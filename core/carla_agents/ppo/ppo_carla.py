import glob
import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from macad_gym.carla.multi_env import MultiCarlaEnv

from core.carla_agents.storage import RolloutStorage
from core.carla_model.model import Policy
from path import Path

from utils.utils import get_scheduler, print_network, mkdirs


class PPOCarla(object):

    def __init__(self, opt, env: MultiCarlaEnv, device: torch.device):
        self.opt = opt
        self.env = env
        self.device = device
        self.train_mode = opt.isTrain
        self.backup_dir = Path(os.path.join(opt.save_path, opt.name))
        mkdirs(self.backup_dir)
        self.start_epoch = 0
        self.model_params = {"hidden_size": 512, "recurrent_input_size": 512}  # parametrize in opt
        self.n_actors = len(self.env.non_auto_actors)

        # Setup modules
        self.model = Policy(env.observation_space, env.action_space, recurrent=True, base_kwargs=self.model_params).to(self.device)
        self.buffer = RolloutStorage(self.opt.rollout_size, self.n_actors, self.env.observation_space, self.env.action_space, self.model_params["recurrent_input_size"]).to(self.device)

        # Load or init
        if not opt.isTrain or opt.continue_train is not None:
            if opt.isTrain:
                self.start_epoch = self.load_model(self.backup_dir, "policy", opt.continue_train)
            else:
                self.load_model(opt.models_path, "policy", opt.model_epoch)
        # else:
        #     init_weights(self.model, init_type="normal")

        if opt.isTrain:
            self.model.train()
            # initialize optimizers
            self.schedulers, self.optimizers = [], []
            self.optimizer_G = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.lr_momentum, opt.lr_beta1), weight_decay=opt.lr_weight_decay)

            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                if self.start_epoch > 0: optimizer.param_groups[0].update({"initial_lr": opt.lr})
                self.schedulers.append(get_scheduler(optimizer, opt, self.start_epoch-1))
        else:
            self.model.eval()
        # print_network(self.model)

        # train cycle params
        self.K_epochs = opt.K_epochs  # 4
        self.batch_size = opt.batch_size  # 16
        # advantages computing
        self.use_gae = True
        self.adv_gamma = opt.adv_gamma  # 0.99
        self.adv_lambda = opt.adv_lambda  # 0.95
        # PPO hyperparams
        self.clip_param = 0.2
        self.value_loss_coef = 0.5  # critic gamma
        self.entropy_coef = 0.01
        # normalization params
        self.max_grad_norm = 0.5  # Max norm of the gradients, weights decay/normalization action
        self.use_clipped_value_loss = False

    def switch_mode(self, mode):
        """
        Change the behaviour of the model.
        Raise an error if the mode specified is not correct.
        :param mode: mode to set. Allowed values are 'train' or 'eval'
        :return: None
        """
        assert(mode in ['train', 'eval'])
        self.train_mode = mode == "train"
        self.model.train() if self.train_mode else self.model.eval()

    def save_model(self, prefix: str = "", model_episode: int = -1):
        """Save the model"""
        save_path = self.backup_dir / "{}_{:04}_model".format(prefix, model_episode)
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, path, prefix, model_episode: int):
        if model_episode == -1:
            load_filename = prefix + '_*_model'
            load_path = Path(sorted(glob.glob(os.path.join(path, load_filename)))[-1])
        else:
            load_filename = prefix + '_{:04}_model'.format(model_episode)
            load_path = path / load_filename
        self.model.load_state_dict(torch.load(load_path))
        epoch = int(load_path.name.split('_')[1])
        print(f"Trained agent ({epoch}) loaded")
        return epoch + 1

    def fill_buffer(self, current_state_obs=None, recurrent_hidden_states=None, output=None, log_prob=None, state_value=None, reward=None, communication_vector=None, termination_mask=None, step=None):
        """
        Add to buffer the data collected from a step interaction with the environment otherwise add first observation or last state value
        """
        assert self.buffer is not None, "You should call `reset_buffer` first."
        assert None not in (current_state_obs, recurrent_hidden_states, output, log_prob, state_value, reward, communication_vector, termination_mask) or \
               ((current_state_obs is not None or state_value is not None) and step >= 0), "The arguments passed do not match any intended behaviour of the method."
        if None not in (current_state_obs, recurrent_hidden_states, output, log_prob, state_value, reward, communication_vector, termination_mask):
            self.buffer.insert(current_state_obs, recurrent_hidden_states, output, log_prob, state_value, reward, communication_vector, termination_mask)
        elif current_state_obs is not None: [self.buffer.obs[key][step].copy_(current_state_obs[key]) for key in current_state_obs]
        else: self.buffer.value_preds[step].copy_(state_value)

    def update(self, valid_steps):
        self.buffer.compute_returns(self.use_gae, self.adv_gamma, self.adv_lambda)

        advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        n_samples = len(advantages)
        for e in range(self.K_epochs):
            if self.model.is_recurrent:
                data_generator = self.buffer.recurrent_generator(advantages, self.n_actors, n_steps=valid_steps+1)
            else:
                data_generator = self.buffer.feed_forward_generator(advantages, self.batch_size, n_steps=valid_steps+1)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, comm_batch, masks_batch, outputs_batch, value_preds_batch, return_batch, old_log_probs_batch, advantages_tgt = sample
                log_probs, values, dist_entropy, _ = self.model.evaluate_actions(obs_batch['img'], obs_batch['msr'], recurrent_hidden_states_batch, comm_batch, masks_batch, outputs_batch)
                action_loss, value_loss = self.ppo_clipped_loss(old_log_probs_batch, log_probs, advantages_tgt, values, return_batch, value_preds_batch)

                self.optimizer_G.zero_grad()
                loss = action_loss + (value_loss * self.value_loss_coef) - (dist_entropy * self.entropy_coef)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer_G.step()

                action_loss_epoch += action_loss.item()
                value_loss_epoch += value_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.K_epochs * n_samples
        action_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        losses = {"value_loss": value_loss_epoch, "action_loss": action_loss_epoch, "dist_entropy": dist_entropy_epoch}

        # Reinitialize buffer
        self.buffer.drain(last_step=valid_steps+1)

        return losses

    def take_action(self, obs, rnn_hxs, comm, mask, deterministic=False):
        return self.model.act(obs['img'], obs['msr'], rnn_hxs, comm, mask, eps=0., deterministic=deterministic)

    def get_value(self, obs, rnn_hxs, comm, masks):
        return self.model.get_value(obs['img'], obs['msr'], rnn_hxs, comm, masks)

    def ppo_clipped_loss(self, old_action_log_probs, newpolicy_probs, adv_targ, values, returns, value_preds):
        ratio = torch.exp((newpolicy_probs - old_action_log_probs).sum(1, keepdim=True))
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds + (values - value_preds).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()  # 0.5 *
        else: value_loss = F.mse_loss(returns, values)  # 0.5 *

        return action_loss, value_loss

    def update_learning_rate(self):
        """
        Update the learning rate value following the schedule instantiated.
        :return: None
        """
        for scheduler in self.schedulers:
           scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
