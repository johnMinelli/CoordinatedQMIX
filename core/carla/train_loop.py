import time
from argparse import Namespace

import gym
import numpy as np
import torch
from macad_gym.carla.multi_env import MultiCarlaEnv

from core.carla.ppo.agents.ppo_carla import PPOCarla
from envs.macad.env import NonSignalizedIntersection4Car
from external.navigation.basic_agent import BasicAgent
from utils.logger import Logger


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


def play_loop(opt: Namespace, env: MultiCarlaEnv, agent: PPOCarla, logger: Logger):
    actors_id = env.non_auto_actors

    for episode in range(agent.start_epoch, opt.episodes):
        for retry in range(5):
            try: current_state_obs = env.reset(); break
            except: time.sleep(0.1)  # retry
        if retry == 5: raise Exception("Multiple failures resetting the environment")
        agent.reset_buffer()

        # setup starting input
        current_state_obs, recurrent_hidden_states, communication_vector, done_masks = preprocess(current_state_obs, env.get_actors_loc(), agent.buffer.recurrent_hidden_states[0], torch.zeros_like(agent.buffer.masks[0]), agent.buffer.masks[0])
        agent.fill_buffer(current_state_obs=current_state_obs, step=0)

        logger.episode_start(episode)
        # total_rewards = {id: 0.0 for id in actors_id}
        done = {"__all__": False}

        # cycle until end of episode
        while not done["__all__"]:
            # fill rollout and update for n times during an episode depending on the siwwze of rollout storage
            for step in range(opt.rollout_size):
                with torch.no_grad():
                    output, log_prob, state_value, next_recurrent_hidden_states = agent.take_action(current_state_obs, recurrent_hidden_states, communication_vector, done_masks)
                if opt.force_comm: output[:,-1] = 1
                actions, comm_mask = output.split(output.size(1)-1, 1)  # just for sake of readability split the output

                try:
                    next_state_obs, reward, done, info = env.step({id: action.detach().cpu().numpy() for id, action in zip(actors_id, actions)},
                                                                  {id: comm_value.detach().cpu().numpy() for id, comm_value in zip(actors_id, comm_mask)})  # the env takes both actions and who communicate
                except Exception as e: print('Failed env step: ' + str(e)); exit()


                reward = torch.FloatTensor(np.fromiter(reward.values(), np.float32)).unsqueeze(-1)
                done_masks = torch.FloatTensor(1 - np.fromiter(done.values(), np.int)[1:]).unsqueeze(-1)

                agent.fill_buffer(current_state_obs, recurrent_hidden_states, output, log_prob, state_value, reward, communication_vector, done_masks)

                # preprocess new state
                current_state_obs, recurrent_hidden_states, communication_vector, done_masks = preprocess(next_state_obs, env.get_actors_loc(), next_recurrent_hidden_states, comm_mask, done_masks)

                if done["__all__"]:
                    break

            # make the update with data collected (ignore if it is just last step)
            if step > 1:
                # add last state value estimate needed for advantage computing and **UPDATE**
                with torch.no_grad():
                    state_value = agent.get_value(current_state_obs, recurrent_hidden_states, communication_vector, done_masks)
                agent.fill_buffer(state_value=state_value, step=step+1)
                # update with batch of data collected
                losses = agent.update(valid_steps=step)
                logger.train_step(step+1, losses, agent.schedulers[0].get_last_lr()[0])
            else:  # be careful with that, you are throwing away last step, but also the reward of done
                agent.buffer.drain(last_step=step+1)
        # optionally save models
        if episode != 0 and (episode % opt.agent_save_interval == 0):
            print(f'Saving agent at episode: {episode}.')
            agent.save_model("policy", episode)

        logger.episode_stop(env.get_stats())
        agent.update_learning_rate()

        if episode != 0 and (episode % opt.agent_valid_interval == 0):
            # TODO maybe
            #  1) create eval env
            #  2) run eval for n episodes (e.g. 10) saving rewards at end
            #  3) log mean results
            pass


def carla_loop(args, device, logger):
    # Create the game environment
    env = NonSignalizedIntersection4Car()

    # Play the game, using the agent
    if not args.shared_multiagent:
        agents = PPOCarla(args, env, device)
        play_loop(args, env, agents, logger)
    env.close()
