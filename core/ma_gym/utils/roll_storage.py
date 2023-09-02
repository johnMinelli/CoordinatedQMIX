import torch


def _flatten_helper(T, N, _tensor):
    if isinstance(_tensor, dict):
        for key in _tensor:
            _tensor[key] = _tensor[key].view(T * N, *(_tensor[key].size()[2:]))
        return _tensor
    else:
        return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    '''
    Class for rollout memory that is collected from the environment at each RL epoch
    The experience that is stored in the rollout memory is used to compute policy gradient in PPO,
    and is thrown away after each epoch since PPO is on-policy
    '''
    def __init__(self, size, n_processes, obs_space, action_space, comm_size):
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

    def put(self, obs, comm, mask, actions, rewards, next_obs, dones):
        self.obs[self._step].copy_(obs)
        if comm is not None: self.comm[self._step].copy_(comm)
        if mask is not None: self.masks[self._step].copy_(mask)
        self.actions[self._step].copy_(actions)
        self.rewards[self._step].copy_(rewards)
        self.next_obs[self._step].copy_(next_obs)
        self.dones[self._step].copy_(dones)

        self._step = (self._step + 1) % self.max_size
        if self._size < self.max_size:
            self._size += 1

    def drain(self):
        self._step = self._size = 0

    def sample_chunk(self, batch_size, chunk_size):
        perm = torch.randperm(self._size - chunk_size - 1)
        obs_batch = []
        comm_batch = []
        masks_batch = []
        actions_batch = []
        rewards_batch = []
        next_obs_batch = []
        dones_batch = []

        for start_ind in perm[:batch_size]:
            end_ind = start_ind + chunk_size + 1
            obs_batch.append(self.obs[start_ind + 1:end_ind])
            comm_batch.append(self.comm[start_ind + 1:end_ind])
            masks_batch.append(self.masks[start_ind + 1:end_ind])
            actions_batch.append(self.actions[start_ind + 1:end_ind])
            rewards_batch.append(self.rewards[start_ind + 1:end_ind])
            next_obs_batch.append(self.next_obs[start_ind + 1:end_ind])
            dones_batch.append(self.dones[start_ind:end_ind])

        # These are all tensors of size (batch_size, chunk_size, -1)
        obs_batch = torch.stack(obs_batch, 0)
        comm_batch = torch.stack(comm_batch, 0)
        masks_batch = torch.stack(masks_batch, 0)
        actions_batch = torch.stack(actions_batch, 0)
        rewards_batch = torch.stack(rewards_batch, 0)
        next_obs_batch = torch.stack(next_obs_batch, 0)
        dones_batch = torch.stack(dones_batch, 0)
        done_masks_batch = torch.ones_like(dones_batch) - dones_batch

        # T, N = batch_size, chunk_size
        # # Flatten the (T, N, ...) tensors to (T * N, ...)
        # obs_batch = _flatten_helper(T, N, obs_batch)
        # next_obs_batch = _flatten_helper(T, N, next_obs_batch)
        # masks_batch = _flatten_helper(T, N, masks_batch)
        # actions_batch = _flatten_helper(T, N, actions_batch)
        # rewards_batch = _flatten_helper(T, N, rewards_batch)
        # dones_batch = _flatten_helper(T, N, dones_batch)

        return obs_batch, comm_batch, masks_batch, actions_batch, rewards_batch, next_obs_batch, done_masks_batch

    def size(self):
        return self._size