import glob
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env
from path import Path

from core.base_agent import BaseAgent
from core.ma_gym.replay_buffer import ReplayBuffer
from utils.utils import print_network, get_scheduler, mkdirs


"""Model network"""
class QNet(nn.Module):
    def __init__(self, agents_ids, observation_space, action_space, model_params):
        super(QNet, self).__init__()

        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.recurrent = model_params["recurrent"]
        self.hx_size = 32
        for i, id in enumerate(agents_ids):
            n_obs = np.prod(observation_space[id].shape)
            setattr(self, 'agent_feature_{}'.format(i), nn.Sequential(nn.Linear(n_obs+action_space[id].n, 64),
                                                                        nn.ReLU(),
                                                                        nn.Linear(64, self.hx_size),
                                                                        nn.ReLU()))
            setattr(self, 'agent_feature_comm_{}'.format(i), nn.Sequential(nn.Linear(n_obs, 64),
                                                                        nn.ReLU(),
                                                                        nn.Linear(32, self.hx_size),
                                                                        nn.ReLU()))
            if self.recurrent:
                setattr(self, 'agent_gru_{}'.format(i), nn.GRUCell(self.hx_size, self.hx_size))
                setattr(self, 'agent_gru_comm_{}'.format(i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(i), nn.Linear(self.hx_size*2, action_space[id].n))
        self.action_dtype = torch.int if action_space[id].__class__.__name__ == 'Discrete' else torch.float32

    def to(self, device):
        self._device = device
        super().to(device)
        return self

    def forward(self, obs, hidden, comm):
        batch_s = obs.shape[0]
        q_values = [torch.empty(batch_s, )] * self.num_agents
        comm_mask = torch.any(comm.view(*comm.shape[:3], -1), -1).unsqueeze(-1)
        n_comm = torch.sum(comm_mask,1)
        comm_hidden = [torch.empty(batch_s, 1, self.hx_size)] * self.num_agents
        next_hidden = [torch.empty(batch_s, 1, self.hx_size)] * self.num_agents
        for i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(i))(obs[:, i, :].reshape(batch_s, -1))
            if self.recurrent:
                if torch.any(n_comm[:,i]):
                    x = getattr(self, 'agent_gru_{}'.format(i))(x, hidden[:, i, :])
                    for j in range(self.num_agents):
                        x_comm = getattr(self, 'agent_feature_comm_{}'.format(i))(comm[:, i, j, :].reshape(batch_s, -1))
                        x_comm = getattr(self, 'agent_gru_comm_{}'.format(i))(x_comm, hidden[:, i, :])
                        comm_hidden[j] = x_comm.unsqueeze(1) * comm_mask[:,i,j]
                    comm_values = torch.cat(comm_hidden,dim=1).sum(1)/torch.masked_fill(n_comm[:,i], n_comm[:,i]==0, 1)
                else: comm_values = torch.zeros_like(x)
                next_hidden[i] = x.unsqueeze(1)
            else:
                for j in range(self.num_agents):
                    if i == j: continue
                    x_comm = getattr(self, 'agent_feature_comm_{}'.format(i))(comm[:, i, j, :].reshape(batch_s, -1))
                    comm_hidden[j] = x_comm.unsqueeze(1)
                comm_values = torch.cat(comm_hidden,dim=1).mean(1)
            hidden = getattr(self, 'agent_q_{}'.format(i))(torch.cat([x, comm_values], dim=1)).unsqueeze(1)

        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, comm, epsilon):
        out, hidden = self.forward(obs, hidden, comm)
        mask = (torch.rand((out.shape[0])) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1]), dtype=self.action_dtype, device=self._device)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).type_as(action)
        action[~mask] = out[~mask].argmax(dim=2).type_as(action)
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size)).to(self._device)


"""Gym agent"""
class VDNGym(BaseAgent):
    """
    Agent for Value decomposition network algorithm
    >> Expect tensors on `device` in input and return tensors on `device` as output
    """

    def __init__(self, opt, env: Env, device: torch.device):

        # Setup modules
        self.q = QNet(env.agents_ids, env.observation_space, env.action_space, model_params=self.model_params).to(device)
        self.q_target = QNet(env.agents_ids, env.observation_space, env.action_space, model_params=self.model_params).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.memory = ReplayBuffer(opt.max_buffer_len, device)

        # Load or init
        self.training = opt.isTrain
        if not opt.isTrain or opt.continue_train is not None:
            if opt.isTrain:
                self.start_epoch = self.load_model(self.backup_dir, "policy", opt.continue_train)
            else:
                self.load_model(opt.models_path, "policy", opt.model_epoch)
        # else:
        #     init_weights(self.model, init_type="normal")

        if opt.isTrain:
            self.q.train()
            # initialize optimizers
            self.schedulers, self.optimizers = [], []
            self.optimizer_G = optim.Adam(self.q.parameters(), lr=opt.lr)

            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                if self.start_epoch > 0: optimizer.param_groups[0].update({"initial_lr": opt.lr})
                self.schedulers.append(get_scheduler(optimizer, opt, self.start_epoch-1))
        else:
            self.q.eval()
        print_network(self.q)

        # train cycle params
        self.K_epochs = opt.K_epochs  # 10
        self.batch_size = opt.batch_size
        self.warm_up_steps = opt.min_buffer_len
        self.gamma = opt.gamma
        self.chunk_size = opt.chunk_size  # 10
        self.grad_clip_norm = opt.grad_clip_norm  # 5
        self.temperature = 1
        assert self.warm_up_steps > (self.batch_size * self.chunk_size) * 2, "Set a the min buffer length to be greater then `(batch_size x chunk_size)x2` to avoid overlappings"

    def init_hidden(self):
        return self.q.init_hidden()

    def init_comm_msgs(self):
        return self.q.init_comm_msgs()

    def switch_mode(self, mode):
        """
        Change the behaviour of the model.
        Raise an error if the mode specified is not correct.
        :param mode: mode to set. Allowed values are 'train' or 'eval'
        :return: None
        """
        assert(mode in ['train', 'eval'])
        self.training = mode == "train"
        self.q.train() if self.training else self.q.eval()

    def save_model(self, prefix: str = "", model_episode: int = -1):
        """Save the model"""
        save_path = self.backup_dir / "{}_{:04}_model".format(prefix, model_episode)
        torch.save(self.q_target.state_dict(), save_path)

    def load_model(self, path, prefix, model_episode: int):
        if model_episode == -1:
            load_filename = prefix + '_*_model'
            load_path = Path(sorted(glob.glob(os.path.join(path, load_filename)))[-1])
        else:
            load_filename = prefix + '_{:04}_model'.format(model_episode)
            load_path = path / load_filename
        self.q.load_state_dict(torch.load(load_path))
        self.q_target.load_state_dict(self.q.state_dict())

        epoch = int(load_path.name.split('_')[1])
        print(f"Trained agent ({epoch}) loaded")
        return epoch + 1

    def decay_exploration(self, episode):
        self.epsilon = max(self.opt.min_epsilon, self.opt.max_epsilon - (self.opt.max_epsilon - self.opt.min_epsilon) * (episode / (0.6 * self.opt.episodes)))

    def take_action(self, observation, hidden=None, dones=None):
        actions, hidden = self.q.sample_action(observation, hidden, 1-dones, self.epsilon if self.training else self.opt.min_epsilon)

        return actions, hidden

    def step(self, state, action, reward, next_state, done):
        self.memory.put((state, action, np.array(reward).tolist(), next_state, np.array(done, dtype=int).tolist()))
        losses = {}

        if self.memory.size() > self.warm_up_steps:
            q_loss_epoch = 0

            _chunk_size = self.chunk_size
            for _ in range(self.K_epochs):
                states, actions, rewards, next_states, done_masks = self.memory.sample_chunk(self.batch_size, _chunk_size)
                hidden = self.q.init_hidden(self.batch_size)
                target_hidden = self.q_target.init_hidden(self.batch_size)

                loss = 0.
                for step_i in range(_chunk_size):
                    with torch.no_grad() if step_i == 0 else nullcontext():
                        # reset recurrent info if the chunk contains a restart (all dones)
                        all_done = torch.all((1 - done_masks[:, step_i]).bool(), -1)
                        hidden[all_done] = self.q.init_hidden(len(hidden[all_done]))
                        target_hidden[all_done] = self.q_target.init_hidden(len(target_hidden[all_done]))

                        q_out, hidden = self.q(states[:, step_i], hidden)
                        q_a = q_out.gather(2, actions[:, step_i].unsqueeze(-1).long()).squeeze(-1)
                        sum_q = q_a.sum(dim=1, keepdims=True)

                        q_target, target_hidden = self.q_target(next_states[:, step_i], target_hidden.detach())
                        max_q_target = q_target.max(dim=2)[0].squeeze(-1)
                        target = (rewards[:, step_i] * done_masks[:, step_i]).sum(dim=1, keepdims=True) + (self.gamma * (max_q_target * done_masks[:, step_i]).sum(dim=1, keepdims=True))

                        loss += F.smooth_l1_loss(sum_q, target.detach())

                self.optimizer_G.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip_norm, norm_type=2)
                self.optimizer_G.step()

                q_loss_epoch += loss.item()

            num_updates = self.K_epochs * _chunk_size
            q_loss_epoch /= num_updates
            losses = {"q_loss": q_loss_epoch}

        return losses

    def update_target_net(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def update_learning_rate(self):
        """
        Update the learning rate value following the schedule instantiated.
        :return: None
        """
        if self.memory.size() > self.warm_up_steps:
            for scheduler in self.schedulers:
               scheduler.step()
            lr = self.optimizers[0].param_groups[0]['lr']
            print('learning rate = %.7f' % lr)
