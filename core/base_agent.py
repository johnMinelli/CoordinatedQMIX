import os

import torch
from gym import Env
from path import Path

from utils.utils import mkdirs


class BaseAgent(object):

    def __init__(self, opt, env: Env, device: torch.device):
        self.opt = opt
        self.env = env
        self.device = device
        self.train_mode = opt.isTrain
        self.n_agents = len(self.env.agents_ids)
        self.start_epoch = 0

    @property
    def learning(self):
        pass

    def init_hidden(self):
        pass

    def init_comm_msgs(self):
        pass

    def switch_mode(self, mode):
        """
        Change the behaviour of the model.
        Raise an error if the mode specified is not correct.
        :param mode: mode to set. Allowed values are 'train' or 'eval'
        :return: None
        """
        assert(mode in ['train', 'eval'])
        self.train_mode = mode == "train"

    def save_model(self, prefix: str = "", model_episode: int = -1):
        """Save the model"""
        pass

    def load_model(self, path, prefix, model_episode: int):
        """Load the weights of the model"""
        pass

    def decay_exploration(self, episode):
        pass

    def take_action(self, observation, hidden=None, comm=None, dones=None, explore=True):
        pass

    def step(self, state, add_in, action, reward, next_state, done):
        pass

    def update_target_net(self):
        pass

    def update_learning_rate(self):
        """
        Update the learning rate value following the schedule instantiated.
        :return: None
        """
        pass
