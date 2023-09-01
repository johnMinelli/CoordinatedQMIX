import gym
import numpy as np
import pygame

class MaGymEnvWrap(object):
    """Wrapper used to for ma-gym based environments"""

    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        self.n_agents_dummy = env.n_agents_dummy
        self.agents_ids = env.agent_ids
        self.clock = pygame.time.Clock()
        self.observation_space = {k: self.env.observation_space[i] for i, k in enumerate(self.agents_ids)}
        self.action_space = {k: self.env.action_space[i] for i, k in enumerate(self.agents_ids)}
        self.no_op = 4

    def set_no_op(self, action):
        self.no_op = action

    def reset(self, options=None):
        obs, _ = self.env.reset(options=options)
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


class CarlaEnvWrap(object):
    """Wrapper used to test CoMix with CARLA AD simulator (using carla-gym interface)"""
    def __init__(self, env):
        self.env = env
        self.n_agents = env.max_num_agents
        self.agents_ids = env.possible_agents
        self.clock = pygame.time.Clock()
        self.observation_space = env.observation_spaces
        self.action_space = env.action_spaces
        self.no_op = 8

    def set_no_op(self, action):
        self.no_op = action

    def reset(self):
        obs_dict = self.env.reset()
        rewards_dict = {id: 0 for id in self.agents_ids}
        dones_dict = {id: False for id in self.agents_ids}
        info_dict = {}
        return obs_dict, rewards_dict, dones_dict, info_dict

    def step(self, actions):
        obs_dict, rewards_dict, term_dict, trunc_dict, _ = self.env.step({id: int(a) for a, id in zip(actions, self.agents_ids)})
        done_dict = {a: (te or tr) for a,te,tr in zip(self.agents_ids, term_dict.values(), trunc_dict.values())}
        info = {}
        info.update({"success": 0})
        return obs_dict, rewards_dict, done_dict, info

    def close(self):
        self.env.close()

    def get_success_metric(self):
        return 0

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
import gym
import numpy as np
import pygame

class MaGymEnvWrap(object):
    """Wrapper used to for ma-gym based environments"""

    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        self.n_agents_dummy = env.n_agents_dummy
        self.agents_ids = env.agent_ids
        self.clock = pygame.time.Clock()
        self.observation_space = {k: self.env.observation_space[i] for i, k in enumerate(self.agents_ids)}
        self.action_space = {k: self.env.action_space[i] for i, k in enumerate(self.agents_ids)}
        self.no_op = 4

    def set_no_op(self, action):
        self.no_op = action

    def reset(self, options=None):
        obs, _ = self.env.reset(options=options)
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


class CarlaEnvWrap(object):
    """Wrapper used to test CoMix with CARLA AD simulator (using carla-gym interface)"""
    def __init__(self, env):
        self.env = env
        self.n_agents = env.max_num_agents
        self.agents_ids = env.possible_agents
        self.clock = pygame.time.Clock()
        self.observation_space = env.observation_spaces
        self.action_space = env.action_spaces
        self.no_op = 8

    def set_no_op(self, action):
        self.no_op = action

    def reset(self):
        obs_dict = self.env.reset()
        rewards_dict = {id: 0 for id in self.agents_ids}
        dones_dict = {id: False for id in self.agents_ids}
        info_dict = {}
        return obs_dict, rewards_dict, dones_dict, info_dict

    def step(self, actions):
        obs_dict, rewards_dict, term_dict, trunc_dict, _ = self.env.step({id: int(a) for a, id in zip(actions, self.agents_ids)})
        done_dict = {a: (te or tr) for a,te,tr in zip(self.agents_ids, term_dict.values(), trunc_dict.values())}
        info = {}
        info.update({"success": 0})
        return obs_dict, rewards_dict, done_dict, info

    def close(self):
        self.env.close()

    def get_success_metric(self):
        return 0

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
