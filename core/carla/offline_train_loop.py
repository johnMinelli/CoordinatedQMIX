import time
from argparse import Namespace
from typing import Callable

import gym
import numpy as np
import torch
# from macad_gym.carla.multi_env import MultiCarlaEnv

from core.carla.ppo.agents.ppo_carla import PPOCarla
from envs.macad.env import NonSignalizedIntersection4Car
from envs.multiprocessing_env import SubprocVecEnv
from external.navigation.basic_agent import BasicAgent
from utils.logger import Logger

global _device


def preprocess(obs, actors_coords, rnn_hxs, comm_mask, done_mask, avg_comm=True):
    """
    The communication matrix built has the following logic:
    - row by row you see who is alive and has enabled the communication to speak with other actors
    - column by columns you see from who, the current actor, will receive communications
    mask the hidden state matrix, scale, and sum to obtain the communication vector
    """
    processed_obs = {}
    for key in list(list(obs.values())[0].keys()):
        processed_obs[key] = torch.tensor(np.array([_obs[key] for _obs in obs.values()])).type_as(rnn_hxs)
    done_mask = done_mask.to(rnn_hxs.device)  # TODO the scaling must be regulated in a 0-1 range not by distance overall otherwise the number are tooo small
    eye_mask = torch.logical_not(torch.eye(rnn_hxs.size(0))).unsqueeze(-1).to(rnn_hxs.device)
    # construct the communication vector as sum of hidden vectors of others actors scaled by relative distance
    coords = torch.tensor(np.array(list(actors_coords.values())))[:,:-1].type_as(rnn_hxs)
    coords = coords.expand(coords.size(0), *coords.size())
    # torch.cdist(a, a.transpose(0, 1), p=2).T[0]
    l2_dist_coords = (torch.triu((coords - coords.transpose(0, 1)).pow(2).sum(2).sqrt()) + torch.tril(torch.ones_like(coords)[:,:,0])).unsqueeze(-1)  # put 1s in 0 pos to do division
    distance_mask = (l2_dist_coords < 100)
    rnn_hxs_scaled = torch.triu(rnn_hxs.expand(rnn_hxs.size(0), *rnn_hxs.size()).permute(2,0,1)).permute(1,2,0) * distance_mask * eye_mask / l2_dist_coords
    # mask with termination mask and predicted will of communicate (the receiver must hear). Also remove 
    mask = (done_mask * done_mask.T).unsqueeze(-1) * comm_mask.unsqueeze(-1) * (distance_mask * distance_mask.transpose(0, 1)) * eye_mask
    rnn_hxs_scaled_masked = (rnn_hxs_scaled + rnn_hxs_scaled.transpose(0, 1)) * mask
    if avg_comm:  # scale by number of contributing actors for network’s sake
        actors_communicating = (mask.sum(0, keepdim=True)).expand(*rnn_hxs_scaled_masked.size())
        div_mask = actors_communicating > 0
        rnn_hxs_scaled_masked[div_mask] = rnn_hxs_scaled_masked[div_mask] / actors_communicating[div_mask]
    comm = rnn_hxs_scaled_masked.sum(0)

    return processed_obs, rnn_hxs, comm, done_mask


def play_loop(opt: Namespace, env_fn: Callable[[], MultiCarlaEnv], agents_fn: Callable[[Namespace, EnvWrap], object], logger: Logger):
    # Initialize elements
    env = env_fn()
    agents = agents_fn(opt, env)

    agents_ids = env.non_auto_actors

    for episode in range(agents.start_epoch, opt.episodes):
        logger.episode_start(episode)
        step = 0

        agents.decay_exploration(episode)
        hxs = agents.init_hidden()
        if isinstance(hxs, tuple): hxs, g_hxs = hxs
        comm_msgs = agents.init_comm_msgs()
        for retry in range(5):
            try: current_state_obs = env.reset(); break
            except: time.sleep(0.1)  # retry
        if retry == 5: raise Exception("Multiple failures resetting the environment")
        current_state_obs = np.expand_dims(np.stack([current_state_obs[id] for id in agents_ids]), 0)
        dones = np.expand_dims(np.stack([dones[id] for id in agents_ids]), 0)

        # total_rewards = {id: 0.0 for id in actors_id}
        done = {"__all__": False}

        # cycle until end of episode
        while not done["__all__"]:
            step += 1
            # select agent actions from observations
            output, log_prob, state_value, next_recurrent_hidden_states = agents.take_action(torch.Tensor(current_state_obs).to(_device), recurrent_hidden_states, communication_vector, torch.Tensor(dones).to(_device))
            if isinstance(actions, tuple): actions, hxs, comm_msgs = actions
            if isinstance(hxs, tuple): hxs, g_hxs = hxs
            actions = actions.cpu().squeeze(0).numpy()

            try:
                next_state_obs, rewards_dict, dones, info = env.step({id: action.detach().cpu().numpy() for id, action in zip(agents_ids, actions)})
            except Exception as e: print('Failed env step: ' + str(e)); exit()

            rewards = np.array(list(rewards_dict.values()))

            # preprocess new state
            # rewards = np.expand_dims(rewards, 0)
            # next_state_obs = np.expand_dims(np.stack([next_state_obs[id] for id in agents_ids]), 0)
            # dones = np.expand_dims(np.stack([dones[id] for id in agents_ids]), 0)
            current_state_obs, recurrent_hidden_states, communication_vector, done_masks = preprocess(next_state_obs, env.get_actors_loc(), next_recurrent_hidden_states, comm_mask, done_masks)
            # let agent take step and add to memory
            losses = agents.step(current_state_obs, actions, rewards, next_state_obs, dones)
            logger.train_step(1, {**rewards_dict, **losses}, agents.schedulers[0].get_last_lr()[0])

        # optionally save models
        if episode != 0 and (episode % opt.agent_save_interval == 0):
            print(f'Saving agent at episode: {episode}.')
            agents.save_model("policy", episode)

        logger.episode_stop(env.get_stats())
        agents.update_learning_rate()

        # Update
        if episode != 0 and (episode % opt.update_target_interval == 0):
            agents.update_target_net()

        if episode != 0 and (episode % opt.agent_valid_interval == 0):
            # TODO maybe
            #  1) create eval env
            #  2) run eval for n episodes (e.g. 10) saving rewards at end
            #  3) log mean results
            pass
            #     logger.valid_step(1, {})
            # logger.valid_stop({"rewards": {k: v for k, v in zip(test_env.agents_ids, total_rewards)}},
            #                   {"evaders": env.get_opponent_num()})
            # agents.switch_mode('train')
    env.close()

def carla_loop(args, device, logger):
    global _device
    _device = device

    def create_env_fn():
        # Create the game environment
        env = NonSignalizedIntersection4Car()
        return env

    def create_agent_fn(opt, env):
        agent = PPOCarla(args, env, device)
        return agent

    proc_env_fn = create_env_fn if args.parallel_envs <= 1 else SubprocVecEnv([create_env_fn for _ in range(args.parallel_envs)])

    # run loop
    play_loop(args, proc_env_fn, create_agent_fn, logger)
