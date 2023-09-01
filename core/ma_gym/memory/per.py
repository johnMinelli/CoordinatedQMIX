import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gym.spaces import Dict


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros((2 * capacity - 1))
        # self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, index):
        index = index + self.capacity - 1
        delta = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, delta)

    def _propagate(self, index, delta):
        parent = (index - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def get_leaf(self, value):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_index = parent
                break
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_index = leaf_index - self.capacity + 1
        return data_index, self.tree[leaf_index]

    def total_priority(self):
        return self.tree[0]


class PrioritizedStorage(object):
    '''
    Class for rollout memory that is collected from the environment at each RL epoch
    The experience that is stored in the rollout memory is used to compute policy gradient in PPO,
    and is thrown away after each epoch since PPO is on-policy
    '''
    def __init__(self, size, n_processes, obs_space, action_space, comm_size, eps=1e-6):
        obs_space = list(obs_space.values())[0]
        self.obs = torch.zeros(size, n_processes, *(obs_space.shape))
        self.next_obs = torch.zeros(size, n_processes, *(obs_space.shape))
        self.comm = torch.zeros(size, n_processes, comm_size)
        self.masks = torch.zeros(size, n_processes, n_processes, 1)
        action_space = list(action_space.values())[0]
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = torch.zeros(size, n_processes, 1).long()
        else:
            self.actions = torch.zeros(size, n_processes, action_space.shape[0])
        self.rewards = torch.zeros(size, n_processes, 1)
        self.dones = torch.zeros(size, n_processes, 1)
        self.buffer = SumTree(size)
        self.epsilon = eps  # even the 0 priority sample should have at some probability of being sampled

        self.max_size = size
        self.n_processes = n_processes
        self._step = self._size = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.comm = self.comm.to(device)
        self.masks = self.masks.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_obs = self.next_obs.to(device)
        self.dones = self.dones.to(device)
        return self

    def put(self, priority, state):
        
        self.buffer.add(priority.cpu().numpy()+self.epsilon, self._step)
        
        obs, comm, mask, actions, rewards, next_obs, dones = state
        self.obs[self._step].copy_(obs)
        self.comm[self._step].copy_(comm)
        self.masks[self._step].copy_(mask)
        self.actions[self._step].copy_(actions)
        self.rewards[self._step].copy_(rewards)
        self.next_obs[self._step].copy_(next_obs)
        self.dones[self._step].copy_(dones)

        self._step = (self._step+1) % self.max_size
        if self._size < self.max_size:
            self._size += 1

    def drain(self):
        self._step = self._size = 0
        self.buffer = SumTree(self.max_size)

    def sample_chunk(self, batch_size, chunk_size):
        def check_distrib():
            return [torch.sum(self.rewards[self.buffer.get_leaf(np.random.uniform(segment * i, segment * (i - 1)))[0]]) for i in range(batch_size)]

        total_priority = self.buffer.total_priority()
        segment = total_priority / batch_size

        indices = []
        priorities = []

        obs_batch, comm_batch, masks_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = [], [], [], [], [], [], []
        i=0
        while len(indices) != batch_size:
            # Get indexes in segment distribution
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            index, priority = self.buffer.get_leaf(value)
            # Cycle
            if i<batch_size:
                i+=1
            else:
                if len(indices) == 0:
                    raise Exception(f"Not enough data to sample sequences of size {chunk_size}.")
            # Get data from index set
            if index > chunk_size-1:
                indices.append(index)
                priorities.append(priority / total_priority)

                start_ind = index - chunk_size
                end_ind = index + 1
                obs_batch.append(self.obs[start_ind+1:end_ind])
                comm_batch.append(self.comm[start_ind+1:end_ind])
                masks_batch.append(self.masks[start_ind+1:end_ind])
                actions_batch.append(self.actions[start_ind+1:end_ind])
                rewards_batch.append(self.rewards[start_ind+1:end_ind])
                next_obs_batch.append(self.next_obs[start_ind+1:end_ind])
                dones_batch.append(self.dones[start_ind:end_ind])

        obs_batch = torch.stack(obs_batch, 0)  # (batch, chunk, proc, -1)
        comm_batch = torch.stack(comm_batch, 0)
        masks_batch = torch.stack(masks_batch, 0)
        actions_batch = torch.stack(actions_batch, 0)
        rewards_batch = torch.stack(rewards_batch, 0)
        next_obs_batch = torch.stack(next_obs_batch, 0)
        dones_batch = torch.stack(dones_batch, 0)
        done_masks_batch = torch.ones_like(dones_batch)-dones_batch

        return obs_batch, comm_batch, masks_batch, actions_batch, rewards_batch, next_obs_batch, done_masks_batch

    def size(self):
        return self._size