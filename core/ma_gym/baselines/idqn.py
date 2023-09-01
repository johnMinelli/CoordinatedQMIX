import collections
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env
from path import Path

from core.ma_gym.memory.replay_buffer import ReplayBuffer
from utils.utils import mkdirs, get_scheduler, print_network


"""Model network"""
class QNet(nn.Module):
    def __init__(self, agents_ids, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        assert not recurrent, "This algorithm is not implemented with recurrent functionalities, set the `recurrent` flag at `False`"

        self.num_agents = len(agents_ids)
        self.hx_size = 64
        for i, id in enumerate(agents_ids):
            n_obs = np.prod(observation_space[id].shape)
            setattr(self, 'agent_{}'.format(i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU()))
            setattr(self, 'agent_comm_{}'.format(i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU()))
            setattr(self, 'agent_q_{}'.format(i), nn.Linear(self.hx_size*2, action_space[id].n))
        self.action_dtype = torch.int if action_space[id].__class__.__name__ == 'Discrete' else torch.float32

    def forward(self, obs, hidden, comm):
        batch_s = obs.shape[0]
        q_values = [torch.empty(batch_s, )] * self.num_agents
        comm_hidden = [torch.empty(batch_s, 1, self.hx_size)] * (self.num_agents-1)
        for i in range(self.num_agents):
            x = getattr(self, 'agent_{}'.format(i))(obs[:, i, :].reshape(batch_s, -1)).unsqueeze(1)
            for j in range(self.num_agents):
                if i == j: continue
                x_comm = getattr(self, 'agent_comm_{}'.format(i))(comm[:, i, j, :].reshape(batch_s, -1))
                comm_hidden[j] = x_comm.unsqueeze(1)
            comm_values = torch.cat(comm_hidden, dim=1).mean(1)
            q_values[i] = getattr(self, 'agent_q_{}'.format(i))(torch.cat([x, comm_values], dim=1)).unsqueeze(1)
        return torch.cat(q_values, dim=1)

    def sample_action(self, obs, hidden, comm, epsilon):
        out = self.forward(obs, hidden, comm).detach().cpu()
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1]), dtype=self.action_dtype)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).type_as(action)
        action[~mask] = out[~mask].argmax(dim=2).type_as(action)
        return action

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))


"""Gym agent"""
class IDQNGym(object):

    def __init__(self, opt, env: Env, device: torch.device):
        self.opt = opt
        self.env = env
        self.device = device
        self.train_mode = opt.isTrain
        self.backup_dir = Path(os.path.join(opt.save_path, opt.name))
        mkdirs(self.backup_dir)
        self.start_epoch = 0

        # Setup modules
        self.q = QNet(env.agents_ids, env.observation_space, env.action_space, self.opt.recurrent).to(device)
        self.q_target = QNet(env.agents_ids, env.observation_space, env.action_space, self.opt.recurrent).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.memory = ReplayBuffer(opt.max_buffer_len, device)

        # Load or init
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
            self.optimizer_G = optim.Adam(self.q.parameters(), lr=opt.lr, betas=(opt.lr_momentum, opt.lr_beta1), weight_decay=opt.lr_weight_decay)

            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                if self.start_epoch > 0: optimizer.param_groups[0].update({"initial_lr": opt.lr})
                self.schedulers.append(get_scheduler(optimizer, opt, self.start_epoch-1))
        else:
            self.q.eval()
        print_network(self.q)

        # train cycle params
        self.K_epochs = opt.K_epochs
        self.batch_size = opt.batch_size
        self.warm_up_steps = opt.min_buffer_len
        self.gamma = opt.gamma
        self.epsilon = 1

    def init_hidden(self):
        return None

    def switch_mode(self, mode):
        """
        Change the behaviour of the model.
        Raise an error if the mode specified is not correct.
        :param mode: mode to set. Allowed values are 'train' or 'eval'
        :return: None
        """
        assert(mode in ['train', 'eval'])
        self.train_mode = mode == "train"
        self.q.train() if self.train_mode else self.q.eval()

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
        self.epsilon = max(self.opt.min_epsilon, self.opt.max_epsilon - (self.opt.max_epsilon - self.opt.min_epsilon) * (episode / (0.4 * self.opt.episodes)))
        # self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon) * np.exp(-(self.t - self.burnin_steps) / self.epsilon_decay_rate)

    def take_action(self, observation, hidden=None, comm=None, explore=True):
        return self.q.sample_action(observation, hidden, comm, self.epsilon if explore else 0)

    def step(self, state, comm, action, reward, next_state, next_comm, done):
        self.memory.put((state, comm, action, (np.array(reward)).tolist(), next_state, next_comm, np.array(done, dtype=int).tolist()))
        losses = {}
        # TODO sarebbe da modificare con i chunk anche se sono tanti agenti indipendenti agiscono comunque con recurrent modules
        if self.memory.size() > self.warm_up_steps:
            q_loss_epoch = 0

            for _ in range(self.K_epochs):
                states, comm, actions, rewards, next_states, next_comm, dones = self.memory.sample(self.batch_size)
                hidden = self.q.init_hidden(self.batch_size).to(self.device)
                target_hidden = self.q_target.init_hidden(self.batch_size).to(self.device)

                q_out = self.q(states, hidden, comm)
                q_a = q_out.gather(2, actions.unsqueeze(-1).long()).squeeze(-1)
                max_q_target = self.q_target(next_states, target_hidden, next_comm).max(dim=2)[0]
                target = rewards + self.gamma * max_q_target * dones
                loss = F.smooth_l1_loss(q_a, target.detach())

                self.optimizer_G.zero_grad()
                loss.backward()
                self.optimizer_G.step()

                q_loss_epoch += loss.item()

            num_updates = self.K_epochs * self.batch_size
            q_loss_epoch /= num_updates
            losses = {"value_loss": q_loss_epoch}

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
