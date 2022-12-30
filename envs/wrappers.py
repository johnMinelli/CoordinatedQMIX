import gym
import numpy as np
import pygame
import math
from pettingzoo.utils import BaseWrapper


class MaGymEnvWrap(object):
    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        self.agents_ids = [str(k) for k in env.agent_pos.keys()]
        self.clock = pygame.time.Clock()
        self.observation_space = {k:gym.spaces.box.Box(np.array([0.,0.,0.,0.]),np.array([1.,1.,1.,1.]),(4,)) for i, k in enumerate(self.agents_ids)}
        self.action_space = {k: env.action_space[i] for i, k in enumerate(self.agents_ids)}

    def reset(self):
        obs = self.env.reset()
        obs_dict = {id: o[:-1]+[round(f[0] / (self.env._grid_shape[0] - 1), 2), round(f[1] / (self.env._grid_shape[1] - 1), 2)] for o, f, id in zip(obs, self.env.final_agent_pos.values(), self.agents_ids)}
        rewards_dict = {id: 0 for id in self.agents_ids}
        dones_dict = {id: False for id in self.agents_ids}
        info_dict = []
        return obs_dict, rewards_dict, dones_dict, info_dict

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        obs_dict = {id: o[:-1]+[round(f[0] / (self.env._grid_shape[0] - 1), 2), round(f[1] / (self.env._grid_shape[1] - 1), 2)] for o, f, id in zip(obs, self.env.final_agent_pos.values(), self.agents_ids)}
        rewards_dict = {id: r for r, id in zip(rewards, self.agents_ids)}
        dones_dict = {id: d for d, id in zip(dones, self.agents_ids)}
        info_dict = info
        return obs_dict, rewards_dict, dones_dict, info_dict

    def close(self):
        self.env.close()

    def get_near_matrix(self):
        agents = self.unwrapped.env.agents
        positions = [a.current_pos for a in agents]
        size_obs = list(self.observation_space.values())[0].shape[0]
        identity = np.expand_dims(np.identity(len(agents)), -1) * size_obs
        near_mat = np.all((np.abs(np.expand_dims(positions, 0) - np.expand_dims(positions, 1)) + identity) <= math.floor(size_obs / 2), 2)
        return near_mat

    def get_success_metric(self):
        return int(self.env._step_count < self.env._max_steps)

    def render(self, **kwargs):
        self.clock.tick(30)
        self.env.render(**kwargs)
        # pygame.display.flip()

    def _register_input(self):
        if pygame.get_init() and pygame.display.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Stop")
                    return


class EnvWrap(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = env.unwrapped.num_agents
        self.agents_ids = env.unwrapped.agents
        self.clock = pygame.time.Clock()
        self.observation_space = {k: self.observation_space(k) for k in self.agents_ids}
        self.action_space = {k: self.action_space(k) for k in self.agents_ids}

    def reset(self, **kwargs):
        if not pygame.get_init():
            pygame.init()
        super().reset(**kwargs)
        obs_dict = {id: self.observe(id).copy() for id in self.agents_ids}
        rewards_dict = self.rewards
        dones_dict = {id: self.terminations[id] or self.truncations[id] for id in self.agents_ids}
        info_dict = self.infos
        return obs_dict, rewards_dict, dones_dict, info_dict

    def step(self, actions):
        self._register_input()
        for id, action in zip(self.agents_ids, actions):
            if self.agent_selection != id:
                raise Exception(f"Action selected do not match the current agent: current agent is {self.agent_selection}, but action is for {id}") 
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

    def get_success_metric(self):
        return self.unwrapped.env.n_evaders-len(self.unwrapped.env.evaders)

    def render(self, **kwargs):
        self.clock.tick(30)
        ret = super().render(**kwargs)
        pygame.display.flip()
        return ret

    def _register_input(self):
        if pygame.get_init() and pygame.display.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Stop")
                    return
