import glob
import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env
from path import Path

from core.base_agent import BaseAgent
from core.carla.ppo.model.utils import init
from core.ma_gym.comix.comix_agent import QPolicy
from core.ma_gym.memory.roll_storage import RolloutStorage
from utils.utils import print_network, get_scheduler, mkdirs


"""Centralized critic"""
class CriticNet(nn.Module):
    def __init__(self, agents_ids, observation_space, action_space, model_params=None):
        super(CriticNet, self).__init__()
        if model_params is None:
            model_params = {}
        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.agents_ids = agents_ids
        self.obs_shape = list(observation_space.values())[0].shape
        self.obs_size = np.prod(self.obs_shape)
        self.action_space = action_space
        _to_ma = lambda m, args: nn.ModuleList([m(*args) for _ in range(self.num_agents)])
        _init_fc_norm = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0))

        self.tot_action_size = sum([a_space.n for a_space in self.action_space.values()])
        self.tot_obs_size = sum([np.prod(o_space.shape) for o_space in observation_space.values()])
        self.critic = nn.ModuleList([nn.Sequential(_init_fc_norm(nn.Linear(self.tot_obs_size + self.tot_action_size, 128)),
                                                                    nn.ReLU(),
                                                                    _init_fc_norm(nn.Linear(128, 64)),
                                                                    nn.ReLU(),
                                                                    _init_fc_norm(nn.Linear(64, 1))) for _ in agents_ids])

    def forward(self, obs, action):
        batch_s = obs.shape[0]
        q_values = []
        x = torch.cat((obs.reshape(batch_s, -1), action.reshape(batch_s, -1)), dim=1)
        for i in range(self.num_agents):
            q_values.append(self.critic[i](x).unsqueeze(1))
        return torch.cat(q_values, dim=1)  # TO DO


"""Gym agent"""
class CoordMADDPGGymAgent(BaseAgent):
    """
    Agent for Multi agent deep deterministic policy gradient algorithm
    >> Expect tensors on `device` in input and return tensors on `device` as output
    """

    def __init__(self, opt: Namespace, env: Env, device: torch.device):
        super().__init__(opt, env, device)
        self.q_params = {"eval_coord_mask": opt.coord_mask_type, "ae_comm": bool(opt.ae_comm), "hidden_size": opt.hi, "coord_recurrent_size": opt.hc, "ae_comm_size": 16, "input_proc_size": opt.hs, "cnn_input_proc": bool(opt.cnn_input_proc)}
        self.mixer_params = {"hidden_size": opt.hm}

        # Setup multi agent modules (dimension with n agents)
        self.q_policy = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.q_params).to(device)
        self.q_policy_target = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.q_params).to(device)
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        # no need of having a target for mask predictor module
        self.q_policy_target.ma_coordinator = self.q_policy.ma_coordinator
        self.critic = CriticNet(env.agents_ids, env.observation_space, env.action_space).to(device)
        self.critic_target = CriticNet(env.agents_ids, env.observation_space, env.action_space).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.memory = RolloutStorage(opt.max_buffer_len, env.n_agents, env.observation_space, env.action_space, self.q_policy.plan_size).to(device)

        # Load or init
        self.training = opt.isTrain
        if not opt.isTrain or opt.continue_train is not None:
            if opt.isTrain:
                self.start_epoch = self.load_model(self.backup_dir, "policy", opt.continue_train)
            else:
                self.load_model(opt.models_path, "policy", opt.model_epoch)
        # else:
        #     init_weights(self.model, init_type="normal")

        self.q_policy_target.eval()
        self.critic_target.eval()
        if opt.isTrain:
            self.q_policy.train()
            self.critic.train()
            # initialize optimizers
            self.schedulers, self.optimizers, self.initial_lr = [], [], []

            self.optimizer_policy = optim.Adam(self.q_policy.get_policy_parameters(), lr=opt.lr_q, weight_decay=opt.lr_weight_decay)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=opt.lr_c)
            self.optimizer_coordinator = optim.Adam(self.q_policy.get_coordinator_parameters(), lr=opt.lr_co)

            self.optimizers.append(self.optimizer_policy)
            self.initial_lr.append(opt.lr_q)
            self.optimizers.append(self.optimizer_critic)
            self.initial_lr.append(opt.lr_c)
            self.optimizers.append(self.optimizer_coordinator)
            self.initial_lr.append(opt.lr_co)
            if self.q_policy.shared_comm_ae:
                self.optimizer_ae = optim.Adam(self.q_policy.ae.parameters(), lr=opt.lr_ae)
                self.optimizers.append(self.optimizer_ae)
                self.initial_lr.append(opt.lr_ae)

            for i, optimizer in enumerate(self.optimizers):
                if self.start_epoch > 0: optimizer.param_groups[0].update({"initial_lr": self.initial_lr[i]})
                self.schedulers.append(get_scheduler(optimizer, opt, self.start_epoch-1))
        else:
            self.q_policy.eval()
        print_network(self.q_policy)
        print_network(self.critic)

        # train cycle params
        self.no_op = env.no_op
        self.K_epochs = opt.K_epochs  # 0.02
        self.coord_K_epochs = opt.coord_K_epochs  # 0.1
        self.batch_size = opt.batch_size
        self.warm_up_steps = opt.min_buffer_len
        self.gamma = opt.gamma
        self.chunk_size = opt.chunk_size  # 3,10
        self.grad_clip_norm = opt.grad_clip_norm  # 5
        self.lambda_coord = opt.lambda_coord
        self.lambda_q = opt.lambda_q/self.chunk_size
        self.tau = opt.tau
        self.coord_loss = torch.zeros(1, device=device)
        self.loss_coordinator = torch.zeros(1, device=device)
        self.ae_loss = torch.zeros(1, device=device)
        self.coord_stats = {}
        self.updates = 0
        self.coord_updates = 0

        # assert self.warm_up_steps > (self.batch_size * self.chunk_size) * 2, "Set a the min buffer length to be greater then `(batch_size x chunk_size)x2` to avoid overlappings"

    @property
    def learning(self):
        return self.memory.size() >= self.warm_up_steps

    def init_hidden(self):
        return self.q_policy.init_hidden()

    def switch_mode(self, mode):
        """
        Change the behaviour of the model.
        Raise an error if the mode specified is not correct.
        :param mode: mode to set. Allowed values are 'train' or 'eval'
        :return: None
        """
        assert(mode in ['train', 'eval'])
        self.training = mode == "train"
        self.q_policy.train() if self.training else self.q_policy.eval()

    def save_model(self, prefix: str = "", model_episode: int = -1):
        """Save the model""" # TO DO
        save_path_c = self.backup_dir / "{}_critic_{:04}_model".format(prefix, model_episode)
        torch.save(self.critic.state_dict(), save_path_c)
        save_path_q = self.backup_dir / "{}_q_{:04}_model".format(prefix, model_episode)
        torch.save(self.q_policy.state_dict(), save_path_q)
        return {"policy": save_path_q, "critic": save_path_c}

    def load_model(self, path, prefix, model_episode: int):
        if model_episode == -1:
            load_filename_c = prefix + '_critic_*_model'
            load_filename_q = prefix + '_q_*_model'
            load_path_c = Path(sorted(glob.glob(os.path.join(path, load_filename_c)))[-1])
            load_path_q = Path(sorted(glob.glob(os.path.join(path, load_filename_q)))[-1])
        else:
            load_filename_c = prefix + '_critic_{:04}_model'.format(model_episode)
            load_filename_q = prefix + '_q_{:04}_model'.format(model_episode)
            load_path_c = path / load_filename_c
            load_path_q = path / load_filename_q
        self.critic.load_state_dict(torch.load(load_path_c))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.q_policy.load_state_dict(torch.load(load_path_q))
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        epoch = int(load_path_c.name.split('_')[2])
        print(f"Trained agent ({epoch}) loaded")
        return epoch + 1

    def decay_exploration(self, episode):
        self.epsilon = max(self.opt.gumbel_min_temp, self.opt.gumbel_max_temp - (self.opt.gumbel_max_temp - self.opt.gumbel_min_temp) * (episode / (0.6 * self.opt.episodes)))

    def _soft_update(self):
        for param_target, param in zip(self.q_policy_target.get_policy_parameters(), self.q_policy.get_policy_parameters()):
            param_target.data.copy_((param_target.data * (1.0 - self.tau)) + (param.data * self.tau))
        for param_target, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            param_target.data.copy_((param_target.data * (1.0 - self.tau)) + (param.data * self.tau))

    def take_action(self, observation, hidden, dones):
        observation = observation.unsqueeze(0)
        hidden, coordinator_hidden = hidden
        dones_mask = (1-dones).unsqueeze(0)
        co_q_input = None
        if not self.training:
            actions, hidden, coordinator_hidden = self.q_policy.take_action(observation, hidden, coordinator_hidden, dones_mask, self.opt.gumbel_min_temp)
        else:
            q_out, hidden, coordinator_hidden, inv_q_out, coord_masks, co_q_input, ae_loss = self.q_policy(observation, hidden, coordinator_hidden, dones_mask, eval_coord=True)  # explore both actions and coordination
            actions = self.q_policy.sample_action_from_qs(q_out, self.epsilon).squeeze().detach()

            if self.memory.size() >= self.warm_up_steps:  # avoid coordinator optimization while Q is still dumb
                # optimize the coordinator while acting using the max q differences given by a mask respect its inverse
                dones_mask = dones_mask.squeeze(-1)
                pred_q_s = (q_out.max(dim=-1)[0] * dones_mask).detach()
                coord_masks = coord_masks.transpose(2, 0)  # F.softmax(coord_masks.transpose(2, 0), dim=-1)
                if self.q_policy.eval_coord_mask == "optout":
                    inv_pred_q_s = (inv_q_out.max(dim=-1)[0] * dones_mask.unsqueeze(-1) * dones_mask.unsqueeze(-2)).detach()
                    self.loss_coordinator += torch.sum(torch.sum(
                        nn.ReLU()(inv_pred_q_s - pred_q_s.unsqueeze(-1).expand(inv_pred_q_s.shape)).unsqueeze(-1).expand(coord_masks.shape) * coord_masks, -1))
                else:
                    inv_pred_q_s = (inv_q_out.max(dim=-1)[0] * dones_mask).detach()
                    self.loss_coordinator += torch.sum(torch.sum(torch.sum(
                        nn.ReLU()(inv_pred_q_s - pred_q_s).unsqueeze(-1).unsqueeze(-1).expand(coord_masks.shape) * coord_masks, -1), -1) / self.n_agents)

                self.coord_updates += self.coord_K_epochs
                if self.coord_updates >= 1:
                    self.optimizer_coordinator.zero_grad()
                    (self.loss_coordinator * self.coord_K_epochs).backward()
                    self.optimizer_coordinator.step()
                    self.coord_loss += self.loss_coordinator
                    self.coord_stats.update({"no": (coord_masks[:, :, :, 0]>coord_masks[:, :, :, 1]).sum().item(), "yes": (coord_masks[:, :, :, 0]<coord_masks[:, :, :, 1]).sum().item(), "good": ((inv_pred_q_s-pred_q_s.expand(inv_pred_q_s.shape))<=0).sum().item(), "bad": ((inv_pred_q_s-pred_q_s.expand(inv_pred_q_s.shape))>0).sum().item(), "agree": (inv_q_out.view(q_out.size(0),q_out.size(1),-1).argmax(2)%q_out.size(2) == q_out.argmax(dim=2)).sum().item()})
                    self.coord_updates -= 1
                    self.loss_coordinator = torch.zeros(1, device=self.loss_coordinator.device)

            # optimize the ae while acting
            if self.q_policy.shared_comm_ae:
                self.optimizer_ae.zero_grad()
                ae_loss.backward()
                self.optimizer_ae.step()
                self.ae_loss += ae_loss

        return actions.unsqueeze(-1).detach(), (hidden.detach(), coordinator_hidden.detach()), co_q_input

    def step(self, state, add_in, action, reward, next_state, done):
        self.memory.put(state, *([a.squeeze(0) for a in add_in] if add_in is not None else [None, None]), action, reward, next_state, done)

        # drain ea loss got from action in order for logging
        losses = {"ae_loss": self.ae_loss.item(), "coord_loss": self.coord_loss.item()}
        losses.update(self.coord_stats)
        self.ae_loss, self.coord_loss = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        num_updates = 0

        if self.memory.size() >= self.warm_up_steps:
            self.updates += self.K_epochs
            if self.updates >= 1:
                critic_loss_epoch = 0
                q_loss_epoch = 0
                num_updates = int(self.updates)
                for _ in range(num_updates):
                    states, comm, coord_masks, actions, rewards, next_states, done_masks = self.memory.sample_chunk(self.batch_size, self.chunk_size)

                    hidden, coord_hidden = self.q_policy.init_hidden(self.batch_size)
                    target_hidden, target_coord_hidden = self.q_policy_target.init_hidden(self.batch_size)

                    for step_i in range(self.chunk_size):
                        # reset recurrent info /*if the chunk contains a restart (all dones)*/ if done
                        reset = (1-done_masks[:, step_i]).squeeze(-1).bool()
                        next_reset = (1-done_masks[:, step_i+1]).squeeze(-1).bool()
                        m, cm, tm, tcm = *self.q_policy.init_hidden(len(reset)), *self.q_policy_target.init_hidden(len(next_reset))
                        hidden[reset], coord_hidden.permute(2,0,1,3)[reset] = m[reset], cm.permute(2,0,1,3)[reset]
                        target_hidden[reset], target_coord_hidden.permute(2,0,1,3)[reset] = tm[reset], tcm.permute(2,0,1,3)[reset]
                        target_hidden[next_reset], target_coord_hidden.permute(2,0,1,3)[next_reset] = tm[next_reset], tcm.permute(2,0,1,3)[next_reset]
                        done_masks[:, step_i][torch.all((1-done_masks[:, step_i]).squeeze(-1).bool(),-1)] = 1  # fresh start when needed

                        next_q_target, target_hidden, target_coord_hidden, _, _, _, _ = self.q_policy_target(next_states[:, step_i], target_hidden.detach(), target_coord_hidden.detach(), done_masks[:, step_i+1])  # act at your best of coord
                        next_q_probs = F.gumbel_softmax(next_q_target.view(-1, next_q_target.size(-1)), tau=0.1, hard=True).view(*next_q_target.shape)
                        q_target = ((rewards[:, step_i] * done_masks[:, step_i]) + self.gamma * (self.critic_target(next_states[:, step_i], next_q_probs) * done_masks[:, step_i+1])).squeeze(-1)

                        action_one_hot = torch.zeros_like(next_q_probs).scatter_(-1, actions[:, step_i].type(torch.int64), 1.)
                        pred_q = self.critic(states[:, step_i], action_one_hot).squeeze(-1)
                        critic_loss = F.smooth_l1_loss(pred_q, q_target.detach())

                        self.optimizer_critic.zero_grad()
                        critic_loss.backward()
                        self.optimizer_critic.step()

                        q, hidden, coord_hidden, _, _, _, _ = self.q_policy(states[:, step_i], hidden.detach(), coord_hidden.detach(), done_masks[:, step_i])  # here what coord should I use?
                        q_probs = F.gumbel_softmax(q.view(-1, q.size(-1)), tau=0.1, hard=True).view(*q.shape)
                        policy_loss = -self.critic(states[:, step_i], q_probs).mean()

                        self.optimizer_policy.zero_grad()
                        policy_loss.backward()
                        self.optimizer_policy.step()

                        q_loss_epoch += policy_loss.item()
                        critic_loss_epoch += critic_loss.item()

                    self._soft_update()  # update ogni 100 steps?

                num_updates = self.K_epochs
                q_loss_epoch /= num_updates
                critic_loss_epoch /= num_updates
                losses.update({"q_loss": q_loss_epoch, "critic_loss": critic_loss_epoch})

        return num_updates, losses

    def update_learning_rate(self):
        """
        Update the learning rate value following the schedule instantiated.
        :return: None
        """
        if self.memory.size() > self.warm_up_steps:
            for scheduler in self.schedulers:
               scheduler.step()
            lr_mu = self.optimizers[0].param_groups[0]['lr']
            lr_q = self.optimizers[1].param_groups[0]['lr']
            print('learning rate mu = %.7f, learning rate q = %.7f' % (lr_mu, lr_q))
