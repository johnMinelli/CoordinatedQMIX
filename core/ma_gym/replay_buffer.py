import collections
import random

import numpy as np
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler


class ReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.states = collections.deque(maxlen=buffer_limit)
        self.comm = collections.deque(maxlen=buffer_limit)
        self.actions = collections.deque(maxlen=buffer_limit)
        self.rewards = collections.deque(maxlen=buffer_limit)
        self.next_states = collections.deque(maxlen=buffer_limit)
        self.next_comm = collections.deque(maxlen=buffer_limit)
        self.dones = collections.deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition):
        """ Adds new transition to the replay buffer """
        assert len(transition) == 7, "The memory transition do not contain the right elements in order to be stored."
        self.states.append(transition[0])
        self.comm.append(transition[1])
        self.actions.append(transition[2])
        self.rewards.append(transition[3])
        self.next_states.append(transition[4])
        self.next_comm.append(transition[5])
        self.dones.append(transition[6])

    def sample(self, batch_size):
        """ Sample batch from replay buffer """
        indexes = np.random.randint(0, len(self.states) - 1, batch_size)
        mini_batch_states = np.array(self.states[indexes])
        mini_batch_comm = np.array(self.comm[indexes])
        mini_batch_actions = np.array(self.actions[indexes])
        mini_batch_rewards = np.array(self.rewards[indexes])
        mini_batch_next_states = np.array(self.next_states[indexes])
        mini_batch_next_comm = np.array(self.next_comm[indexes])
        mini_batch_dones = np.array(self.dones[indexes])
        mini_batch_dones = (np.ones(mini_batch_dones.shape) - mini_batch_dones)

        return torch.tensor(mini_batch_states, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_comm, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_actions, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_rewards, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_next_states, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_next_comm, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_dones, dtype=torch.float).squeeze(1).to(self.device)

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.states) - chunk_size, batch_size)

        chunk_states = np.stack([np.array(self.states)[idx:idx + chunk_size] for idx in start_idx])
        chunk_comm = np.stack([np.array(self.comm)[idx:idx + chunk_size] for idx in start_idx])
        chunk_actions = np.stack([np.array(self.actions)[idx:idx + chunk_size] for idx in start_idx])
        chunk_rewards = np.stack([np.array(self.rewards)[idx:idx + chunk_size] for idx in start_idx])
        chunk_next_states = np.stack([np.array(self.next_states)[idx:idx + chunk_size] for idx in start_idx])
        chunk_next_comm = np.stack([np.array(self.next_comm)[idx:idx + chunk_size] for idx in start_idx])
        chunk_dones = np.stack([np.array(self.dones)[idx:idx + chunk_size] for idx in start_idx])
        chunk_dones = (np.ones(chunk_dones.shape) - chunk_dones)

        return torch.tensor(chunk_states, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_comm, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_actions, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_rewards, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_next_states, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_next_comm, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_dones, dtype=torch.float).squeeze(2).to(self.device)

    def size(self):
        return len(self.states)
