import gym
import numpy as np
import pygame

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
        obs_dict = {id: o for o, id in zip(obs, self.agents_ids)}
        rewards_dict = {id: r for r, id in zip(rewards, self.agents_ids)}
        dones_dict = {id: d for d, id in zip(dones, self.agents_ids)}
        info.update({"success": np.all(dones) and self.env.step_count < self.env.max_steps})
        return obs_dict, rewards_dict, dones_dict, info

    def close(self):
        self.env.close()

    def get_success_metric(self):
        return self.env.success

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
