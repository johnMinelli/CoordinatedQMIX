import math
import time
from argparse import Namespace
from typing import Callable
import sys
import gym
import numpy as np
import torch

from core.base_agent import BaseAgent
from core.ma_gym.comix.comix_agent import CoordQMixGymAgent
from core.ma_gym.comix.covdn_agent import CoordVDNGymAgent
from core.ma_gym.baselines.idqn import IDQNGym
from core.ma_gym.comix.comaddpg_agent import CoordMADDPGGymAgent
from core.ma_gym.baselines.qmix import QMixGym
from envs.wrappers import MaGymEnvWrap
from utils.logger import Logger

global _device


def play_loop(opt: Namespace, env_fn: Callable[[], MaGymEnvWrap], agents_fn: Callable[[Namespace, MaGymEnvWrap], BaseAgent], logger: Logger):
    # Initialize elements
    env = env_fn()
    test_env = env_fn()
    agents = agents_fn(opt, env)
    step = 0

    agents_ids = env.agents_ids

    # logger.episode_start(opt.episodes-1)
    # logger.train_step(12700, {}, {})
    # logger.episode_stop({"rewards": {}}, {"success": 0})

    for episode in range(agents.start_epoch, opt.episodes):
        warmup_ended = agents.learning
        if warmup_ended: logger.episode_start(episode)
        total_rewards = np.zeros(env.n_agents)

        agents.decay_exploration(episode)
        hxs = agents.init_hidden()
        current_state_obs, _, dones, _ = env.reset()
        current_state_obs = torch.Tensor(np.stack([current_state_obs[id] for id in agents_ids])).to(_device)
        dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)
        # cycle until end of episode
        while not dones.all():
            step += 1
            # select agent actions from observations
            actions, hxs, add_in = agents.take_action(current_state_obs, hxs, dones)
            actions[dones.squeeze().bool()] = env.no_op  # TODO change it with None

            # env step
            next_state_obs, rewards_dict, dones, info = env.step(list(map(lambda a: None if dones[a[0]] else a[1][0].cpu().numpy(), enumerate(actions))))
            if opt.render_mode == "human": env.render()

            rewards = np.array(list(rewards_dict.values()))
            total_rewards += rewards

            # preprocess new state
            rewards = torch.Tensor(np.expand_dims(rewards, -1)).to(_device)
            next_state_obs = torch.Tensor(np.stack([next_state_obs[id] for id in agents_ids])).to(_device)
            dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)

            # let agent store it and take a learning take step (if `warmup_ended`); skip if ended by max_step
            n_updates, losses = agents.step(current_state_obs, add_in, actions, rewards, next_state_obs, dones)
            if warmup_ended:
                n_log_steps = n_updates if opt.log_per_update else 1
                logger.train_step(n_log_steps, {**rewards_dict, **losses}, agents.schedulers[0].get_last_lr()[0])
            # update for next iteration
            current_state_obs = next_state_obs

        if warmup_ended:
            # optionally save models
            if episode != 0 and (episode % opt.agent_save_interval == 0):
                print(f'Saving agent at episode: {episode}.')
                agents.save_model("policy", episode)

            logger.episode_stop({"rewards": {k: v for k,v in zip(env.agents_ids, total_rewards)}}, {"success": env.get_success_metric()})
            agents.update_learning_rate()

            # Hard update (if implemented)
            if step >= opt.update_target_interval:
                step = 0
                agents.update_target_net()
            # for comparison eval each training episode using as step the number of updates done in the training episode
            if episode != 0 and (episode % opt.agent_valid_interval == 0):
                test_env = env_fn()
                agents.switch_mode('eval')
                total_rewards = np.zeros(test_env.n_agents)
                success = 0

                logger.valid_start()
                for episode in range(opt.val_episodes):

                    hxs = agents.init_hidden()
                    current_state_obs, _, dones, _ = test_env.reset(options={"test": True})
                    current_state_obs = torch.Tensor(np.stack([current_state_obs[id] for id in agents_ids])).to(_device)
                    dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)
                    while not dones.all():

                        actions, hxs, _ = agents.take_action(current_state_obs, hxs, dones)
                        next_state_obs, rewards_dict, dones, info = test_env.step(list(map(lambda a: None if dones[a[0]] else a[1][0].cpu().numpy(), enumerate(actions))))
                        if opt.render_mode == "human" or opt.render_mode == "human_val": test_env.render()
                        total_rewards += np.array(list(rewards_dict.values()))
                        next_state_obs = torch.Tensor(np.stack([next_state_obs[id] for id in agents_ids])).to(_device)
                        dones = torch.Tensor(np.expand_dims(np.stack([dones[id] for id in agents_ids]), -1)).to(_device)

                        current_state_obs = next_state_obs
                    logger.valid_step(1, {})
                    success += test_env.get_success_metric()
                test_env.close()
                logger.valid_stop({"rewards": {k: v for k,v in zip(test_env.agents_ids, total_rewards)}}, {"success": success/opt.val_episodes})
                agents.switch_mode('train')
    env.close()

    print(f'Saving agent at episode: {opt.episodes}.')
    models = agents.save_model("policy", opt.episodes)
    for name, path in models.items():
        logger.log_artifact(path, name, f"checkpoint_{opt.episodes}")


def gym_loop(args: Namespace, device: torch.device, logger: Logger):
    global _device
    _device = device

    def create_env_fn():
        # Create the game environment; the env name is filtered by parser
        if args.env == "switch_dev" or args.env == "CoMix_switch":
            gym.envs.register(id='CustomSwitch4-v0', entry_point='envs.switch:Switch', kwargs=
            {'n_agents': 4, 'full_observable': False, 'step_cost': 0.0, 'max_steps': 500})
            env = MaGymEnvWrap(gym.make("CustomSwitch4-v0"))
            args.rew_threshold = 0.01
            args.min_buffer_len, args.max_buffer_len, args.chunk_size, args.lambda_q, args.update_target_interval = 1000, 20000, 1, 10, 40000
        elif args.env == "predator_prey_dev" or "CoMix_predator_prey" in args.env:
            xy, n = (14, 8) if args.env == "CoMix_predator_prey_8" else (16, 16) if args.env == "CoMix_predator_prey_16" else (12, 4)
            gym.envs.register(id='CustomPredatorPrey-v0', entry_point='envs.predator_prey:PredatorPrey', kwargs=
            {'grid_shape': (xy, xy), 'n_agents': n, 'n_preys': 16, 'prey_move_probs': (0.2, 0.2, 0.2, 0.2, 0.2), 'full_observable': False, 'penalty': 0, 'step_cost': 0, 'max_steps': 500, 'agent_view_range': (5, 5)})
            env = MaGymEnvWrap(gym.make("CustomPredatorPrey-v0"))
            args.rew_threshold = 0.001
            args.min_buffer_len, args.max_buffer_len, args.chunk_size, args.chunk_size, args.update_target_interval, args.hc = 5000, 20000, 10, 1, 40000, 128
        elif args.env == "transport_dev" or "CoMix_transport" in args.env:
            n, h = (4, 2) if args.env == "CoMix_transport_2" else (8, 4) if args.env == "CoMix_transport_4" else (2, 1)
            gym.envs.register(id='CustomTransport-v0', entry_point='envs.transport:Transport', kwargs=
            {'grid_size': (16,10), 'n_agents': n, 'n_loads': h, 'full_observable': False, 'step_cost': 0, 'max_steps': 500, 'agent_view_range': (5, 5)})
            env = MaGymEnvWrap(gym.make("CustomTransport-v0"))
            args.rew_threshold = 0.01
            args.min_buffer_len, args.max_buffer_len, args.chunk_size, args.hc = 5000, 10000, 1, 64
            args.lambda_q, args.update_target_interval = (10, 20000) if n == 2 else (15, 30000) if n == 4 else (20, 40000)
        else:
            env = gym.make(args.env)
        return env

    def create_agent_fn(opt: Namespace, env):
        # Choose the agent implementation
        if args.agent == 'idqn':
            agent = IDQNGym(opt, env, _device)
        elif args.agent == 'vdn':
            agent = CoordVDNGymAgent(opt, env, _device)
        elif args.agent == 'qmix':
            agent = QMixGym(opt, env, _device)
        elif args.agent == 'maddpg':
            args.gumbel_max_temp = 10
            args.gumbel_min_temp = 0.1
            agent = CoordMADDPGGymAgent(opt, env, _device)
        else:
            # my implementation
            agent = CoordQMixGymAgent(opt, env, _device)

        return agent

    # run loop
    play_loop(args, create_env_fn, create_agent_fn, logger)
