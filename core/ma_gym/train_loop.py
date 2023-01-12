import math
import time
from argparse import Namespace
from typing import Callable

import gym
import numpy as np
import pygame
import torch
from pettingzoo.utils import BaseWrapper

from core.base_agent import BaseAgent
from core.ma_gym.comix.comix_agent import CoordQMixGymAgent
from core.ma_gym.comix.covdn_agent import CoordVDNGymAgent
from core.ma_gym.idqn import IDQNGym
from core.ma_gym.comix.comaddpg_agent import CoordMADDPGGymAgent
from core.ma_gym.qmix import QMixGym
from core.ma_gym.vdn import VDNGym
from envs.multiprocessing_env import SubprocVecEnv
from envs.wrappers import EnvWrap, MaGymEnvWrap
from utils.logger import Logger

global _device


def play_loop(opt: Namespace, env_fn: Callable[[], EnvWrap], agents_fn: Callable[[Namespace, EnvWrap], BaseAgent], logger: Logger):
    # Initialize elements
    env = env_fn()
    test_env = env_fn()
    agents = agents_fn(opt, env)

    agents_ids = env.agents_ids
    # obs_a, obs_b, obs_c = list(env.observation_space.values())[0].shape
    # expand_near_mask = lambda m: np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.expand_dims(m, 0), -1), obs_a, -1), -1), obs_b, -1), -1), obs_c, -1)

    for episode in range(agents.start_epoch, opt.episodes):
        logger.episode_start(episode)
        total_rewards = np.zeros(env.n_agents)
        step = 0

        agents.decay_exploration(episode)
        hxs = agents.init_hidden()
        current_state_obs, _, dones, _ = env.reset()
        current_state_obs = np.expand_dims(np.stack([current_state_obs[id] for id in agents_ids]), 0)
        dones = np.expand_dims(np.stack([dones[id] for id in agents_ids]), 0)

        # cycle until end of episode
        while not dones.all():
            step += 1
            # select agent actions from observations
            actions, hxs, add_in = agents.take_action(torch.Tensor(current_state_obs).to(_device), hxs, torch.Tensor(dones).to(_device))
            actions = actions.view(-1).cpu().numpy()
            add_in = [a.cpu().numpy() for a in add_in] if add_in is not None else None

            # env step
            next_state_obs, rewards_dict, dones, info = env.step(list(map(lambda a: None if dones[0, a[0]] else a[1], enumerate(actions))))
            if opt.render_mode == "human": env.render()

            rewards = np.array(list(rewards_dict.values()))
            total_rewards += rewards

            # preprocess new state
            rewards = np.expand_dims(rewards, 0)
            next_state_obs = np.expand_dims(np.stack([next_state_obs[id] for id in agents_ids]), 0)
            dones = np.expand_dims(np.stack([dones[id] for id in agents_ids]), 0)
            # let agent take step and add to memory
            losses = agents.step(current_state_obs, add_in, actions, rewards, next_state_obs, dones)
            logger.train_step(1, {**rewards_dict, **losses}, agents.schedulers[0].get_last_lr()[0])
            # update for next iteration
            current_state_obs = next_state_obs

        # optionally save models
        if episode != 0 and (episode % opt.agent_save_interval == 0):
            print(f'Saving agent at episode: {episode}.')
            agents.save_model("policy", episode)

        logger.episode_stop({"rewards": {k: v for k,v in zip(env.agents_ids, total_rewards)}}, {"success": env.get_success_metric()})
        agents.update_learning_rate()

        # Update
        if episode != 0 and (episode % opt.update_target_interval == 0):
            agents.update_target_net()

        if episode != 0 and (episode % opt.agent_valid_interval == 0):
            test_env = env_fn()
            agents.switch_mode('eval')
            total_rewards = np.zeros(test_env.n_agents)
            step = 0
            successes = 0

            logger.valid_start()
            for episode in range(opt.val_episodes):

                hxs = agents.init_hidden()
                current_state_obs, _, dones, _ = test_env.reset()
                current_state_obs = np.expand_dims(np.stack([current_state_obs[id] for id in agents_ids]), 0)
                dones = np.expand_dims(np.stack([dones[id] for id in agents_ids]), 0)
                while not dones.all():
                    step += 1

                    actions, hxs, _ = agents.take_action(torch.Tensor(current_state_obs).to(_device), hxs, torch.Tensor(dones).to(_device))
                    actions = actions.view(-1).cpu().numpy()
                    next_state_obs, rewards_dict, dones, info = test_env.step(list(map(lambda a: None if dones[0, a[0]] else a[1], enumerate(actions))))
                    if opt.render_mode == "human" or opt.render_mode == "human_val": test_env.render()
                    total_rewards += np.array(list(rewards_dict.values()))
                    next_state_obs = np.expand_dims(np.stack([next_state_obs[id] for id in agents_ids]), 0)
                    dones = np.expand_dims(np.stack([dones[id] for id in agents_ids]), 0)

                    current_state_obs = next_state_obs
                logger.valid_step(1, {})
                successes += test_env.get_success_metric()
            test_env.close()
            logger.valid_stop({"rewards": {k: v for k,v in zip(test_env.agents_ids, total_rewards)}}, {"success": successes/opt.val_episodes})
            agents.switch_mode('train')
    env.close()

    print(f'Saving agent at episode: {opt.episodes}.')
    models = agents.save_model("policy", opt.episodes)
    for name, path in models.items():
        logger.log_artifact(path, name, f"checkpoint_{opt.episodes}")

def gym_loop(args, device, logger):
    global _device
    _device = device

    def create_env_fn():
        # Create the game environment
        if args.env == "pursuit":
            from pettingzoo.sisl import pursuit_v4
            env = EnvWrap(pursuit_v4.env(max_cycles=500, x_size=18, y_size=8, shared_reward=False, n_evaders=16, n_pursuers=8,
                           obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01, catch_reward=5.0,
                           urgency_reward=0.0, surround=True, constraint_window=1.0, render_mode=args.render_mode, rand_init=False))
        elif args.env == "switch":
            gym.envs.register(id='MySwitch4-v0', entry_point='ma_gym.envs.switch:Switch', kwargs={'n_agents': 4, 'full_observable': False, 'step_cost': -0.1, 'max_steps': 250})
            env = MaGymEnvWrap(gym.make("ma_gym:MySwitch4-v0"))
        # elif args.env == "matrix":
        #     env = NStepMatrixGame(kwargs={"steps": 10})
        elif args.env == "trafficjunction":
            env = MaGymEnvWrap(gym.make("ma:trafficjunctionv0"))
        else:
            env = gym.make(args.env)
        return env

    def create_agent_fn(opt, env):
        # Set default arguments
        if args.agent == 'qmix':
            args.lr = 0.001
            args.batch_size = 32
            args.gamma = 0.99
            args.grad_clip_norm = 5
            args.episodes = 10000
            args.max_epsilon = 0.9
            args.min_epsilon = 0.1
            args.K_epochs = 10
            args.chunk_size = 10
            args.update_target_interval = 20
            agent = QMixGym(opt, env, _device)
        elif args.agent == 'idqn':
            args.lr = 0.0005
            args.batch_size = 32
            args.gamma = 0.99
            args.max_epsilon = 0.9
            args.min_epsilon = 0.1
            args.K_epochs = 10
            args.update_target_interval = 20
            agent = IDQNGym(opt, env, _device)
        elif args.agent == 'maddpg':
            args.tau = 0.005
            args.gumbel_max_temp = 10
            args.gumbel_min_temp = 0.1
            agent = CoordMADDPGGymAgent(opt, env, _device)
        elif args.agent == 'vdn':
            agent = CoordVDNGymAgent(opt, env, _device)
        else:
            # my implementation
            agent = CoordQMixGymAgent(opt, env, _device)

        return agent

    proc_env_fn = create_env_fn if args.parallel_envs <= 1 else SubprocVecEnv([create_env_fn for _ in range(args.parallel_envs)])

    # run loop
    play_loop(args, proc_env_fn, create_agent_fn, logger)
    # TODO generalize with named tuples to step the agent and fill the replay_memory