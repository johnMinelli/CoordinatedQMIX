import math
import time
from argparse import Namespace
from typing import Callable

import gym
import numpy as np
import pygame
import torch
from pettingzoo.utils import BaseWrapper

from core.ma_gym.idqn import IDQNGym
from core.ma_gym.maddpg import MADDPGGym
from core.ma_gym.qmix import QMixGym
from core.ma_gym.vdn import VDNGym
from envs.multiprocessing_env import SubprocVecEnv
from utils.logger import Logger

global _device


class EnvWrap(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        # TODO create a the observationspace and actionspace from fn
        self.n_agents = env.unwrapped.num_agents
        self.agents_ids = env.unwrapped.agents
        self.clock = pygame.time.Clock()
        self.observation_space = {k: self.observation_space(k) for k in self.agents_ids}
        self.action_space = {k: self.action_space(k) for k in self.agents_ids}

    def reset(self, **kwargs):
        super().reset(**kwargs)
        obs_dict = {id: self.observe(id).copy() for id in self.agents_ids}
        rewards_dict = self.rewards
        dones_dict = {id: self.terminations[id] or self.truncations[id] for id in self.agents_ids}
        info_dict = self.infos
        return obs_dict, rewards_dict, dones_dict, info_dict

    def step(self, actions):
        self._register_input()
        for id, action in zip(self.agents_ids, actions):
            super().step(action)
        obs_dict = {id: self.observe(id).copy() for id in self.agents_ids}
        rewards_dict = self.rewards
        dones_dict = {id: self.terminations[id] or self.truncations[id] for id in self.agents_ids}
        info_dict = self.infos
        return obs_dict, rewards_dict, dones_dict, info_dict

    def get_near_matrix(self):
        agents = self.unwrapped.env.agents
        positions = [a.current_pos for a in agents]
        size_obs = list(self.observation_space.values())[0].shape[0]
        identity = np.expand_dims(np.identity(len(agents)), -1) * size_obs
        near_mat = np.all((np.abs(np.expand_dims(positions, 0) - np.expand_dims(positions, 1)) + identity) <= math.floor(size_obs / 2), 2)
        return near_mat

    def render(self, **kwargs):
        self.clock.tick(30)
        ret = super().render(**kwargs)
        pygame.display.flip()
        return ret

    def _register_input(self):
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Stop")
                    return

def play_loop(opt: Namespace, env_fn: Callable[[], EnvWrap], agents_fn: Callable[[Namespace, EnvWrap], object], logger: Logger):
    # Initialize elements
    env = env_fn()
    agents = agents_fn(opt, env)

    agents_ids = env.agents_ids
    obs_a, obs_b, obs_c = list(env.observation_space.values())[0].shape
    expand_near_mask = lambda m: np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.expand_dims(m, 0), -1), obs_a, -1), -1), obs_b, -1), -1), obs_c, -1)

    for episode in range(agents.start_epoch, opt.episodes):
        logger.episode_start(episode)
        total_rewards = np.zeros(env.n_agents)
        step = 0

        agents.decay_exploration(episode)
        hxs = agents.init_hidden()
        current_state_obs, _, dones, _ = env.reset()
        current_state_obs = np.expand_dims(np.stack([current_state_obs[id] for id in agents_ids]), 0)
        comm_msgs = np.repeat(np.expand_dims(current_state_obs, 1), current_state_obs.shape[1], axis=1) * expand_near_mask(env.get_near_matrix())
        dones = np.expand_dims(np.stack([dones[id] for id in agents_ids]), 0)

        # cycle until end of episode
        while not dones.all():
            step += 1
            # select agent actions from observations
            actions = agents.take_action(torch.Tensor(current_state_obs).to(_device), hxs, torch.Tensor(comm_msgs).to(_device))
            if isinstance(actions, tuple):
                actions, hxs = actions[0].cpu().squeeze(0).numpy(), actions[1]
            else: actions = actions.cpu().squeeze(0).numpy()

            # env step
            next_state_obs, rewards_dict, dones, info = env.step(list(map(lambda a: None if dones[0,a[0]] else a[1], enumerate(actions))))
            if opt.render_mode is not None: env.render()

            rewards = np.array(list(rewards_dict.values()))
            total_rewards += rewards

            # prepare data
            rewards = np.expand_dims(rewards, 0)
            next_state_obs = np.expand_dims(np.stack([next_state_obs[id] for id in agents_ids]), 0)
            # compute the communication matrix and mask it by distance of communication
            next_comm_msgs = np.repeat(np.expand_dims(next_state_obs, 1), next_state_obs.shape[1], axis=1) * expand_near_mask(env.get_near_matrix())
            dones = np.expand_dims(np.stack([np.array(dones[id], dtype=int) for id in agents_ids]), 0)
            # let agent take step and add to memory
            losses = agents.step(current_state_obs, comm_msgs, actions, rewards, next_state_obs, next_comm_msgs, dones)

            current_state_obs = next_state_obs
            comm_msgs = next_comm_msgs
            logger.train_step(step, {**rewards_dict, **losses}, agents.schedulers[0].get_last_lr()[0])

        # optionally save models
        if episode != 0 and (episode % opt.agent_save_interval == 0):
            print(f'Saving agent at episode: {episode}.')
            agents.save_model("policy", episode)

        logger.episode_stop({"rewards": {k: v for k,v in zip(env.agents_ids, total_rewards)}})
        agents.update_learning_rate()

        # Update
        if episode != 0 and (episode % opt.update_target_interval == 0):
            agents.update_target_net()

        if episode != 0 and (episode % opt.agent_valid_interval == 0):
            test_env = env_fn()
            total_rewards = np.zeros(test_env.n_agents)
            step = 0

            for episode in range(opt.val_episodes):
                logger.valid_start()

                hxs = agents.init_hidden()
                current_state_obs, _, dones, _ = test_env.reset()
                current_state_obs = np.expand_dims(np.stack([current_state_obs[id] for id in agents_ids]), 0)
                comm_msgs = np.repeat(np.expand_dims(current_state_obs, 1), current_state_obs.shape[1], axis=1) * expand_near_mask(test_env.get_near_matrix())
                dones = np.expand_dims(np.stack([dones[id] for id in agents_ids]), 0)
                while not dones.all():
                    step += 1

                    actions = agents.take_action(torch.Tensor(current_state_obs).to(_device), hxs.to(_device), torch.Tensor(comm_msgs).to(_device), explore=False)
                    if isinstance(actions, tuple): actions, hxs = actions
                    next_state_obs, rewards_dict, dones, info = test_env.step(list(map(lambda a: None if dones[0, a[0]] else a[1], enumerate(actions.squeeze(0).numpy()))))
                    if opt.render_mode is not None: test_env.render()
                    total_rewards += np.array(list(rewards_dict.values()))
                    next_state_obs = np.expand_dims(np.stack([next_state_obs[id] for id in agents_ids]), 0)
                    next_comm_msgs = np.repeat(np.expand_dims(next_state_obs, 1), next_state_obs.shape[1], axis=1) * expand_near_mask(test_env.get_near_matrix())
                    dones = np.expand_dims(np.stack([np.array(dones[id], dtype=int) for id in agents_ids]), 0)

                    current_state_obs = next_state_obs
                    comm_msgs = next_comm_msgs
                logger.valid_step(step, {})
            logger.valid_stop({"rewards": {k: v for k,v in zip(test_env.agents_ids, total_rewards)}})
    env.close()

def gym_loop(args, device, logger):
    global _device
    _device = device

    def create_env_fn():
        # Create the game environment
        if args.env == "pursuit":
            from pettingzoo.sisl import pursuit_v4
            env = EnvWrap(pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=False, n_evaders=8, n_pursuers=16,
                           obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01, catch_reward=5.0,
                           urgency_reward=-0.1, surround=True, constraint_window=1.0, render_mode=args.render_mode))
        elif args.env == "trafficjunction":
            env = gym.make("ma:trafficjunctionv0")
        else:
            env = gym.make(args.env)
        return env

    def create_agent_fn(opt, env):
        # Set default arguments
        if args.agent == 'maddpg':
            # args.env = 'ma_gym:Switch2-v2'
            args.recurrent = True
            args.lr_mu = 0.0005
            args.lr_q = 0.001
            args.batch_size = 32
            args.tau = 0.005
            args.gamma = 0.99
            args.episodes = 10000
            args.K_epochs = 1
            args.chunk_size = 10
            args.gumbel_max_temp = 10
            args.gumbel_min_temp = 0.1
            args.update_target_interval = 4
            agent = MADDPGGym(opt, env, _device)
        elif args.agent == 'qmix':
            # args.env = 'ma_gym:Checkers-v0'
            args.lr = 0.001
            args.batch_size = 32
            args.gamma = 0.99
            args.episodes = 10000
            args.max_epsilon = 0.9
            args.min_epsilon = 0.1
            args.K_epochs = 10
            args.chunk_size = 10
            args.recurrent = True
            args.update_target_interval = 20
            agent = QMixGym(opt, env, _device)
        elif args.agent == 'vdn':
            # args.env = 'ma_gym:Checkers-v0'
            args.recurrent = True
            args.lr = 0.0001
            args.batch_size = 32
            args.gamma = 0.99
            args.grad_clip_norm = 5
            args.episodes = 500
            args.max_epsilon = 0.9
            args.min_epsilon = 0.1
            args.K_epochs = 4
            args.chunk_size = 10
            args.update_target_interval = 4
            agent = VDNGym(opt, env, _device)
        elif args.agent == 'idqn':
            # args.env = 'ma_gym:Switch2-v2'
            args.lr = 0.0005
            args.batch_size = 32
            args.gamma = 0.99
            args.episodes = 30000
            args.max_epsilon = 0.9
            args.min_epsilon = 0.1
            args.K_epochs = 10
            args.update_target_interval = 20
            agent = IDQNGym(opt, env, _device)
        else: raise Exception("No agent algorthm found with name "+args.agent)
        return agent

    proc_env_fn = create_env_fn if args.parallel_envs <= 1 else SubprocVecEnv([create_env_fn for _ in range(args.parallel_envs)])

    # run loop
    play_loop(args, proc_env_fn, create_agent_fn, logger)
