import collections
import random

import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler


class ReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.rng = default_rng()
        self.states = collections.deque(maxlen=buffer_limit)
        self.additional_inputs1 = collections.deque(maxlen=buffer_limit)
        self.additional_inputs2 = collections.deque(maxlen=buffer_limit)
        self.actions = collections.deque(maxlen=buffer_limit)
        self.rewards = collections.deque(maxlen=buffer_limit)
        self.next_states = collections.deque(maxlen=buffer_limit)
        self.dones = collections.deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition, additional_input):
        """ Adds new transition to the replay buffer """
        assert len(transition) == 5, "The memory transition do not contain the right elements in order to be stored."
        self.states.append(transition[0])
        self.actions.append(transition[1])
        self.rewards.append(transition[2])
        self.next_states.append(transition[3])
        self.dones.append(transition[4])
        # TODO can I cat?
        self.additional_inputs1.append(additional_input[0])
        self.additional_inputs2.append(additional_input[1])

    def sample(self, batch_size):
        """ Sample batch from replay buffer """
        indexes = self.rng.choice(len(self.states) - 1, size=batch_size, replace=False)
        mini_batch_states = np.array(self.states[indexes])
        mini_batch_additional_inputs1 = np.array(self.additional_inputs1[indexes])
        mini_batch_additional_inputs2 = np.array(self.additional_inputs2[indexes])
        mini_batch_actions = np.array(self.actions[indexes])
        mini_batch_rewards = np.array(self.rewards[indexes])
        mini_batch_next_states = np.array(self.next_states[indexes])
        mini_batch_dones = np.array(self.dones[indexes])
        mini_batch_done_masks = (np.ones(mini_batch_dones.shape) - mini_batch_dones)

        return torch.tensor(mini_batch_states, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_additional_inputs1, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_additional_inputs2, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_actions, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_rewards, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_next_states, dtype=torch.float).squeeze(1).to(self.device), \
               torch.tensor(mini_batch_done_masks, dtype=torch.float).squeeze(1).to(self.device)

    def sample_chunk(self, batch_size, chunk_size):

        start_idx = self.rng.choice(len(self.states) - chunk_size, size=batch_size, replace=False)
        # to get the prev dones i'll return sequences of 1+chunk_size

        chunk_states = np.stack([np.array(self.states)[idx+1:idx+1 + chunk_size] for idx in start_idx])
        chunk_additional_inputs1 = np.stack([np.array(self.additional_inputs1)[idx+1:idx+1 + chunk_size] for idx in start_idx])
        chunk_additional_inputs2 = np.stack([np.array(self.additional_inputs2)[idx+1:idx+1 + chunk_size] for idx in start_idx])
        chunk_actions = np.stack([np.array(self.actions)[idx+1:idx+1 + chunk_size] for idx in start_idx])
        chunk_rewards = np.stack([np.array(self.rewards)[idx+1:idx+1 + chunk_size] for idx in start_idx])
        chunk_next_states = np.stack([np.array(self.next_states)[idx+1:idx+1 + chunk_size] for idx in start_idx])
        chunk_dones = np.stack([np.array(self.dones)[idx:idx+1 + chunk_size] for idx in start_idx])
        chunk_done_masks = (np.ones(chunk_dones.shape) - chunk_dones)

        return torch.tensor(chunk_states, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_additional_inputs1, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_additional_inputs2, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_actions, dtype=torch.float).to(self.device), \
               torch.tensor(chunk_rewards, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_next_states, dtype=torch.float).squeeze(2).to(self.device), \
               torch.tensor(chunk_done_masks, dtype=torch.float).squeeze(2).to(self.device)

    def size(self):
        return len(self.states)
