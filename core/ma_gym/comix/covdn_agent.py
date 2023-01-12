import glob
import os
from contextlib import nullcontext

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from gym import Env
from path import Path

from core.base_agent import BaseAgent
from core.ma_gym.comix.comix_agent import CoordQMixGymAgent
from core.ma_gym.comix.comix_modules import *
from core.ma_gym.replay_buffer import ReplayBuffer
from utils.utils import print_network, get_scheduler, mkdirs


class VDNMixer(nn.Module):
    def __init__(self, observation_space):
        super(VDNMixer, self).__init__()
        self.num_agents = len(observation_space)
        self.state_size = sum(np.prod(_.shape) for _ in observation_space.values())

    def forward(self, agent_qs, states, dones_mask):
        return agent_qs.sum(dim=1, keepdims=True)

    def eval_states(self, states):
        return torch.ones(states.size(0), self.num_agents, 1).to(states.device)



"""Gym agent"""
class CoordVDNGymAgent(CoordQMixGymAgent):

    def __init__(self, opt, env: Env, device: torch.device):
        super().__init__(opt, env, device)
        self.mix_net = VDNMixer(env.observation_space).to(device)
        self.mix_net_target = VDNMixer(env.observation_space).to(device)