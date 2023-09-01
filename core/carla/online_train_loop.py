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


# cmd = "nvidia-docker run --rm -e NVIDIA_VISIBLE_DEVICES={} -p {}-{}:2000-2002 {} /bin/bash -c \"sed -i '5i sync' ./CarlaUE4.sh; ./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=10 -carla-settings=\"CarlaSettings.ini\"\"".format(
#     gpu_id, port, port + 2, args.image_name)

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

        for retry in range(5):
            try: current_state_obs = env.reset(); break
            except: time.sleep(0.1)  # retry
        if retry == 5: raise Exception("Multiple failures resetting the environment")
        agents.reset_buffer()

        # setup starting input
        current_state_obs, recurrent_hidden_states, communication_vector, done_masks = preprocess(current_state_obs, env.get_actors_loc(), agents.buffer.recurrent_hidden_states[0], torch.zeros_like(agent.buffer.masks[0]), agents.buffer.masks[0])
        agents.fill_buffer(current_state_obs=current_state_obs, step=0)

        # total_rewards = {id: 0.0 for id in actors_id}
        done = {"__all__": False}

        # cycle until end of episode
        while not done["__all__"]:
            # fill rollout and update for n times during an episode depending on the siwwze of rollout storage
            for step in range(opt.rollout_size):
                with torch.no_grad():
                    output, log_prob, state_value, next_recurrent_hidden_states = agents.take_action(current_state_obs, recurrent_hidden_states, communication_vector, done_masks)
                if opt.force_comm: output[:,-1] = 1
                actions, comm_mask = output.split(output.size(1)-1, 1)  # just for sake of readability split the output

                try:
                    next_state_obs, reward, done, info = env.step({id: action.detach().cpu().numpy() for id, action in zip(agents_ids, actions)},
                                                                  {id: comm_value.detach().cpu().numpy() for id, comm_value in zip(agents_ids, comm_mask)})  # the env takes both actions and who communicate
                except Exception as e: print('Failed env step: ' + str(e)); exit()


                reward = torch.FloatTensor(np.fromiter(reward.values(), np.float32)).unsqueeze(-1)
                done_masks = torch.FloatTensor(1 - np.fromiter(done.values(), np.int)[1:]).unsqueeze(-1)

                agents.fill_buffer(current_state_obs, recurrent_hidden_states, output, log_prob, state_value, reward, communication_vector, done_masks)

                # preprocess new state
                current_state_obs, recurrent_hidden_states, communication_vector, done_masks = preprocess(next_state_obs, env.get_actors_loc(), next_recurrent_hidden_states, comm_mask, done_masks)

                if done["__all__"]:
                    break

            # make the update with data collected (ignore if it is just last step)
            if step > 1:
                # add last state value estimate needed for advantage computing and **UPDATE**
                with torch.no_grad():
                    state_value = agents.get_value(current_state_obs, recurrent_hidden_states, communication_vector, done_masks)
                agents.fill_buffer(state_value=state_value, step=step+1)
                # update with batch of data collected
                losses = agents.update(valid_steps=step)
                logger.train_step(step+1, losses, agents.schedulers[0].get_last_lr()[0])
            else:  # be careful with that, you are throwing away last step, but also the reward of done
                agents.buffer.drain(last_step=step+1)

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
