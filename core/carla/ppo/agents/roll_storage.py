import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gym.spaces import Dict


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
    def __init__(self, n_steps, n_processes, obs_space, action_space, recurrent_hidden_state_size):
        obs_space = list(obs_space.values())[0]

        self.obs = torch.zeros(n_steps + 1, n_processes, *(obs_space.shape))
        self.recurrent_hidden_states = torch.zeros(n_steps + 1, 2, n_processes, recurrent_hidden_state_size)  # a dict of tuple(hidden state, cell state)
        self.rewards = torch.zeros(n_steps, n_processes, 1)
        self.value_preds = torch.zeros(n_steps + 1, n_processes, 1)
        self.returns = torch.zeros(n_steps + 1, n_processes, 1)
        self.log_probs = torch.zeros(n_steps, n_processes, 1)

        action_space = list(action_space.values())[0]
        self.actions = torch.zeros(n_steps, n_processes, 1).long()  # actions + communication
        self.masks = torch.ones(n_steps + 1, n_processes, 1)

        self.n_steps = n_steps
        self.n_processes = n_processes
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.log_probs = self.log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        return self

    def insert(self, obs, recurrent_hidden_states, action, log_prob, value_preds, rewards, mask):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(mask)
        self.step += 1

    def drain(self, steps):
        self.obs[0].copy_(self.obs[steps+1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[steps+1])
        self.masks[0].copy_(self.masks[steps+1])
        self.step = 0

    def compute_returns(self, use_gae, gamma, lamb):
        """ GAE PPO advantage given by delta factor i.e. the TD residual of V with discount gamma.
         It can be considered as an estimate of the advantage of an action 'a' at time t.
        """
        if use_gae:
            # PPO general estimate of returns
            gae = 0
            for step in reversed(range(self.step)):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * lamb * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            # Monte Carlo estimate of returns
            self.returns[-1] = self.value_preds[-1]
            for step in reversed(range(self.step)):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, mini_batch_size, n_steps):
        n_samples, n_processes = self.rewards.size()[0:2]
        batch_size = n_processes * (n_samples - n_steps)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_log_probs_batch = self.log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, value_preds_batch, return_batch, old_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, mini_batch_size, n_steps):
        n_processes = self.rewards.size(1)
        assert n_processes >= mini_batch_size, ("PPO requires the number of processes ({}) to be greater than or equal to the number of PPO mini batches ({}).".format(n_processes, mini_batch_size))
        num_proc_per_batch = mini_batch_size
        perm = torch.randperm(n_processes)
        for start_ind in range(0, n_processes, num_proc_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_log_probs_batch = []
            adv_targ_batch = []

            for offset in range(num_proc_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:n_steps, ind])  # self.n_steps-1
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0, :, ind])
                actions_batch.append(self.actions[:n_steps, ind])  # self.n_steps
                value_preds_batch.append(self.value_preds[:n_steps, ind])  # self.n_steps-1
                return_batch.append(self.returns[:n_steps, ind])  # self.n_steps-1
                masks_batch.append(self.masks[:n_steps, ind])  # self.n_steps-1
                old_log_probs_batch.append(self.log_probs[:n_steps, ind])  # self.n_steps
                adv_targ_batch.append(advantages[:n_steps, ind])

            T, N = n_steps, num_proc_per_batch
            # These are all tensors of size (T, N, -1)

            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_log_probs_batch = torch.stack(old_log_probs_batch, 1)
            adv_targ_batch = torch.stack(adv_targ_batch, 1)

            # States is just a (N, -1) tensor
            obs_batch = torch.stack(obs_batch, 1)
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_log_probs_batch = _flatten_helper(T, N, old_log_probs_batch)
            adv_targ_batch = _flatten_helper(T, N, adv_targ_batch)

            yield obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch, value_preds_batch, return_batch, old_log_probs_batch, adv_targ_batch