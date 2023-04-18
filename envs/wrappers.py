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
        self.observation_space = {k: self.env.observation_space[i] for i, k in enumerate(self.agents_ids)}
        self.action_space = {k: self.env.action_space[i] for i, k in enumerate(self.agents_ids)}
        self.no_op = 4

    def set_no_op(self, action):
        self.no_op = action

    def reset(self):
        obs = self.env.reset()
        obs_dict = {id: o for o, id in zip(obs, self.agents_ids)}
        rewards_dict = {id: 0 for id in self.agents_ids}
        dones_dict = {id: False for id in self.agents_ids}
        info_dict = {}
        return obs_dict, rewards_dict, dones_dict, info_dict

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        rewards = [r for r in rewards]
        obs_dict = {id: o for o, id in zip(obs, self.agents_ids)}
        rewards_dict = {id: r for r, id in zip(rewards, self.agents_ids)}
        dones_dict = {id: d for d, id in zip(dones, self.agents_ids)}
        info.update({"success": np.all(dones) and self.env.step_count < self.env.max_steps})
        return obs_dict, rewards_dict, dones_dict, info

    def close(self):
        self.env.close()

    def get_success_metric(self):
        return self.env.max_steps-self.env.step_count

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


class PZEnvWrap(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = env.unwrapped.num_agents
        self.agents_ids = env.unwrapped.agents
        self.clock = pygame.time.Clock()
        self.observation_space = {k: self.observation_space(k) for k in self.agents_ids}
        self.action_space = {k: self.action_space(k) for k in self.agents_ids}
        self.no_op = 0

    def set_no_op(self, action):
        self.no_op = action

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
        self.infos.update({"success": np.all(list(dones_dict.values())) and len(self.unwrapped.env.evaders)==0})
        return obs_dict, rewards_dict, dones_dict, self.infos

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
