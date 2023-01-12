import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env
from path import Path

from core.ma_gym.replay_buffer import ReplayBuffer
from utils.utils import print_network, get_scheduler, mkdirs

"""Model network"""
class MixNet(nn.Module):
    def __init__(self, observation_space, hidden_dim=32, hx_size=64, recurrent=False):
        super(MixNet, self).__init__()

        self._device = torch.device("cpu")
        state_size = sum([_.shape[0] for _ in observation_space.values()])
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        self.n_agents = len(observation_space)
        self.recurrent = recurrent

        hyper_net_input_size = state_size
        if self.recurrent:
            self.gru = nn.GRUCell(state_size, self.hx_size)
            hyper_net_input_size = self.hx_size
        self.hyper_net_weight_1 = nn.Linear(hyper_net_input_size, self.n_agents * hidden_dim)
        self.hyper_net_weight_2 = nn.Linear(hyper_net_input_size, hidden_dim)

        self.hyper_net_bias_1 = nn.Linear(hyper_net_input_size, hidden_dim)
        self.hyper_net_bias_2 = nn.Sequential(nn.Linear(hyper_net_input_size, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, 1))

    def to(self, device):
        self._device = device
        super().to(device)
        return self

    def forward(self, q_values, observations, hidden):
        batch_size, n_agents, obs_size = observations.shape
        state = observations.view(batch_size, n_agents * obs_size)

        x = state
        if self.recurrent:
            hidden = self.gru(x, hidden)
            x = hidden

        weight_1 = torch.abs(self.hyper_net_weight_1(x))
        weight_1 = weight_1.view(batch_size, self.hidden_dim, n_agents)
        bias_1 = self.hyper_net_bias_1(x).unsqueeze(-1)
        weight_2 = torch.abs(self.hyper_net_weight_2(x))
        bias_2 = self.hyper_net_bias_2(x)

        x = torch.bmm(weight_1, q_values.unsqueeze(-1)) + bias_1
        x = torch.relu(x)
        x = (weight_2.unsqueeze(-1) * x).sum(dim=1) + bias_2
        return x, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.hx_size)).to(self._device)


class QNet(nn.Module):
    def __init__(self, agents_ids, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()

        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.recurrent = recurrent
        self.hx_size = 32
        for i, id in enumerate(agents_ids):
            n_obs = np.prod(observation_space[id].shape)
            setattr(self, 'agent_feature_{}'.format(i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                            nn.ReLU(),
                                                                            nn.Linear(128, self.hx_size),
                                                                            nn.ReLU()))
            setattr(self, 'agent_feature_comm_{}'.format(i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                        nn.ReLU(),
                                                                        nn.Linear(128, self.hx_size),
                                                                        nn.ReLU()))
            if recurrent:
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
        comm_hidden = [torch.empty(batch_s, 1, self.hx_size)] * (self.num_agents-1)
        next_hidden = [torch.empty(batch_s, 1, self.hx_size)] * self.num_agents
        for i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(i))(obs[:, i, :].reshape(batch_s, -1))
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(i))(x, hidden[:, i, :])
                for j in range(self.num_agents):
                    if i == j: continue
                    x_comm = getattr(self, 'agent_feature_comm_{}'.format(i))(comm[:, i, j, :].reshape(batch_s, -1))
                    x_comm = getattr(self, 'agent_gru_comm_{}'.format(i))(x_comm, hidden[:, i, :])
                    comm_hidden[j-1 if i<j else j] = x_comm.unsqueeze(1)
                comm_values = torch.cat(comm_hidden,dim=1).mean(1)
                next_hidden[i] = x.unsqueeze(1)
            else:
                for j in range(self.num_agents):
                    if i == j: continue
                    x_comm = getattr(self, 'agent_feature_comm_{}'.format(i))(comm[:, i, j, :].reshape(batch_s, -1))
                    comm_hidden[j] = x_comm.unsqueeze(1)
                comm_values = torch.cat(comm_hidden,dim=1).mean(1)
            q_values[i] = getattr(self, 'agent_q_{}'.format(i))(torch.cat([x, comm_values], dim=1)).unsqueeze(1)

        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, comm, epsilon):
        out, hidden = map(lambda o: o.detach().cpu(), self.forward(obs, hidden, comm))
        mask = (torch.rand((out.shape[0])) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1]), dtype=self.action_dtype, device=self._device)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).type_as(action)
        action[~mask] = out[~mask].argmax(dim=2).type_as(action)
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size)).to(self._device)


"""Gym agent"""
class QMixGym(object):

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
        self.mix_net = MixNet(env.observation_space, recurrent=self.opt.recurrent).to(device)
        self.mix_net_target = MixNet(env.observation_space, recurrent=self.opt.recurrent).to(device)
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
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
            self.mix_net.train()
            # initialize optimizers
            self.schedulers, self.optimizers = [], []
            policy_net_params = []
            policy_net_params += self.q.parameters()
            policy_net_params += self.mix_net.parameters()
            self.optimizer_G = optim.Adam(policy_net_params, lr=opt.lr)

            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                if self.start_epoch > 0: optimizer.param_groups[0].update({"initial_lr": opt.lr})
                self.schedulers.append(get_scheduler(optimizer, opt, self.start_epoch-1))
        else:
            self.q.eval()
        print_network(self.q)
        print_network(self.mix_net)

        # train cycle params
        self.K_epochs = opt.K_epochs  # 10
        self.batch_size = opt.batch_size
        self.warm_up_steps = opt.min_buffer_len
        self.gamma = opt.gamma
        self.chunk_size = opt.chunk_size  # 10
        self.grad_clip_norm = 5  # 5
        self.temperature = 1

    def init_hidden(self):
        return self.q.init_hidden()

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
        save_path = self.backup_dir / "{}_q_{:04}_model".format(prefix, model_episode)
        torch.save(self.q.state_dict(), save_path)
        save_path = self.backup_dir / "{}_mix_{:04}_model".format(prefix, model_episode)
        torch.save(self.mix_net.state_dict(), save_path)

    def load_model(self, path, prefix, model_episode: int):
        if model_episode == -1:
            load_filename_q = prefix + '_q_*_model'
            load_filename_mix = prefix + '_mix_*_model'
            load_path_q = Path(sorted(glob.glob(os.path.join(path, load_filename_q)))[-1])
            load_path_mix = Path(sorted(glob.glob(os.path.join(path, load_filename_mix)))[-1])
        else:
            load_filename_q = prefix + '_q_{:04}_model'.format(model_episode)
            load_filename_mix = prefix + '_mix_{:04}_model'.format(model_episode)
            load_path_q = path / load_filename_q
            load_path_mix = path / load_filename_mix
        self.q.load_state_dict(torch.load(load_path_q))
        self.q_target.load_state_dict(self.q.state_dict())
        self.mix_net.load_state_dict(torch.load(load_path_mix))
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
        epoch = int(load_path_q.name.split('_')[2])
        print(f"Trained agent ({epoch}) loaded")
        return epoch + 1

    def decay_exploration(self, episode):
        self.epsilon = max(self.opt.min_epsilon, self.opt.max_epsilon - (self.opt.max_epsilon - self.opt.min_epsilon) * (episode / (0.6 * self.opt.episodes)))

    def take_action(self, observation, hidden=None, comm=None, explore=True):
        return self.q.sample_action(observation, hidden, comm, self.epsilon if explore else 0)

    def step(self, state, comm, action, reward, next_state, next_comm, done):
        self.memory.put((state, comm, action, (np.array(reward)).tolist(), next_state, next_comm, np.array(done, dtype=int).tolist()))
        losses = {}

        if self.memory.size() > self.warm_up_steps:
            q_loss_epoch = 0

            _chunk_size = self.chunk_size if self.q.recurrent else 1
            for _ in range(self.K_epochs):
                states, comm, actions, rewards, next_states, next_comm, dones = self.memory.sample_chunk(self.batch_size, _chunk_size)
                hidden = self.q.init_hidden(self.batch_size).to(self.device)
                target_hidden = self.q_target.init_hidden(self.batch_size).to(self.device)
                mix_net_target_hidden = self.mix_net_target.init_hidden(self.batch_size)
                mix_net_hidden = [torch.empty_like(mix_net_target_hidden) for _ in range(_chunk_size + 1)]
                mix_net_hidden[0] = self.mix_net_target.init_hidden(self.batch_size).to(self.device)

                loss = 0
                for step_i in range(_chunk_size):
                    q_out, hidden = self.q(states[:, step_i], hidden, comm[:, step_i])
                    q_a = q_out.gather(2, actions[:, step_i].unsqueeze(-1).long()).squeeze(-1)
                    pred_q, next_mix_net_hidden = self.mix_net(q_a, states[:, step_i], mix_net_hidden[step_i])

                    max_q_target, target_hidden = self.q_target(next_states[:, step_i], target_hidden.detach(), next_comm[:, step_i])
                    max_q_target = max_q_target.max(dim=2)[0].squeeze(-1)
                    next_q_total, mix_net_target_hidden = self.mix_net_target(max_q_target, next_states[:, step_i, :, :],
                                                                               mix_net_target_hidden.detach())
                    target = rewards[:, step_i, :].sum(dim=1, keepdims=True) + (self.gamma * next_q_total * (1 - dones[:, step_i]))
                    loss += F.smooth_l1_loss(pred_q, target.detach())

                    done_mask = dones[:, step_i].squeeze(-1).bool()
                    hidden[done_mask] = self.q.init_hidden(len(hidden[done_mask]))
                    target_hidden[done_mask] = self.q_target.init_hidden(len(target_hidden[done_mask])).to(self.device)
                    mix_net_hidden[step_i + 1][~done_mask] = next_mix_net_hidden[~done_mask]
                    # (len(hidden))[done_mask].view(*hidden[done_mask].shape)
                    mix_net_hidden[step_i + 1][done_mask] = self.mix_net.init_hidden(len(mix_net_hidden[step_i][done_mask])).to(self.device)
                    mix_net_target_hidden[done_mask] = self.mix_net_target.init_hidden(len(mix_net_target_hidden[done_mask])).to(self.device)

                self.optimizer_G.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip_norm, norm_type=2)
                torch.nn.utils.clip_grad_norm_(self.mix_net.parameters(), self.grad_clip_norm, norm_type=2)
                self.optimizer_G.step()

                q_loss_epoch += loss.item()

            num_updates = self.K_epochs * _chunk_size
            q_loss_epoch /= num_updates
            losses = {"value_loss": q_loss_epoch}

        return losses

    def update_target_net(self):
        self.q_target.load_state_dict(self.q.state_dict())
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())

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
