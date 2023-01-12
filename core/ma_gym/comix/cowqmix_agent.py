import glob
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from gym import Env
from path import Path

from core.base_agent import BaseAgent
from core.ma_gym.comix.comix_agent import QPolicy, QMixer
from core.ma_gym.comix.comix_modules import *
from core.ma_gym.replay_buffer import ReplayBuffer
from utils.utils import print_network, get_scheduler, mkdirs
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF


"""Gym agent"""
class CoordQMixGymAgent(BaseAgent):

    def __init__(self, opt, env: Env, device: torch.device):
        super().__init__(opt, env, device)
        self.model_params = {"hidden_size": 32, "coord_recurrent_size": 64, "ae_comm_size": 16, "input_proc_size": 2, "cnn_input_proc": bool(opt.cnn_input_proc), "ae_input_proc": bool(opt.ae_input_proc)}
        self.mixer_params = {"hidden_size": 32}

        # Setup multi agent modules (dimension with n agents)
        self.q_policy = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.model_params).to(device)
        self.q_policy_target = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.model_params).to(device)
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        # no need of having a target for mask predictor module
        self.q_policy_target.ma_coordinator.global_coord_net = self.q_policy.ma_coordinator.global_coord_net
        self.q_policy_target.ma_coordinator.boolean_coordinator = self.q_policy.ma_coordinator.boolean_coordinator
        # central policy version
        self.central_q_policy = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.model_params).to(device)
        self.central_q_policy_target = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.model_params).to(device)
        self.central_q_policy_target.load_state_dict(self.q_policy.state_dict())
        self.central_q_policy_target.ma_coordinator.global_coord_net = self.q_policy.ma_coordinator.global_coord_net
        self.central_q_policy_target.ma_coordinator.boolean_coordinator = self.q_policy.ma_coordinator.boolean_coordinator

        self.mix_net = QMixer(env.observation_space, hidden_size=self.mixer_params["hidden_size"], hypernet_layers=opt.mixer).to(device)
        self.mix_net_target = QMixer(env.observation_space, hidden_size=self.mixer_params["hidden_size"], hypernet_layers=opt.mixer).to(device)
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
        self.central_mix = QMixerCentralFF(env.observation_space, hidden_size=self.mixer_params["hidden_size"], hypernet_layers=opt.mixer)  # Feedforward network that takes state and agent utils as input
        self.central_mix_target = QMixerCentralFF(env.observation_space, hidden_size=self.mixer_params["hidden_size"], hypernet_layers=opt.mixer)
        self.central_mix_target.load_state_dict(self.central_mix.state_dict())
        # self.central_mixer = QMixerCentralAtten(args)
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

        self.q_policy_target.eval()
        self.mix_net_target.eval()
        self.central_q_policy_target.eval()
        self.central_mix_target.eval()
        if opt.isTrain:
            self.q_policy.train()
            self.mix_net.train()
            self.central_q_policy.train()
            self.central_mix.train()
            # initialize optimizers
            self.schedulers, self.optimizers, self.initial_lr = [], [], []
            self.policy_net_params = []
            self.mixer_params = list(self.mix_net.parameters())
            self.policy_net_params += self.q_policy.get_policy_parameters()
            self.policy_net_params += self.central_q_policy.get_policy_parameters()
            self.policy_net_params += self.mix_net.parameters()
            self.policy_net_params += self.central_mix_net.parameters()

            self.optimizer_policy = optim.Adam(self.policy_net_params, lr=opt.lr_q)
            self.optimizer_coordinator = optim.Adam(self.q_policy.get_coordinator_parameters(), lr=opt.lr_co)
            self.optimizers.append(self.optimizer_policy)
            self.initial_lr.append(opt.lr_q)
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
        print_network(self.mix_net)

        # train cycle params
        self.K_epochs = opt.K_epochs  # 10
        self.batch_size = opt.batch_size
        self.warm_up_steps = opt.min_buffer_len
        self.gamma = opt.gamma
        self.chunk_size = opt.chunk_size  # 10
        self.grad_clip_norm = 10  # 5
        self.lambda_coord = 0.1
        self.lambda_qmix_loss
        self.lambda_central_loss
        self.coord_loss = 0
        self.coord_stats = {}
        self.ae_loss = 0
        # assert self.warm_up_steps > (self.batch_size * self.chunk_size) * 2, "Set a the min buffer length to be greater then `(batch_size x chunk_size)x2` to avoid overlappings"

    def init_hidden(self):
        return self.q_policy.init_hidden()

    def init_comm_msgs(self):
        return self.q_policy.init_comm_msgs()

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
        """Save the model"""
        save_path_p = self.backup_dir / "{}_q_{:04}_model".format(prefix, model_episode)
        torch.save(self.q_policy.state_dict(), save_path_p)
        save_path_m = self.backup_dir / "{}_mix_{:04}_model".format(prefix, model_episode)
        torch.save(self.mix_net.state_dict(), save_path_m)
        save_path_cp = self.backup_dir / "{}_cq_{:04}_model".format(prefix, model_episode)
        torch.save(self.q_policy.state_dict(), save_path_cp)
        save_path_cm = self.backup_dir / "{}_cmix_{:04}_model".format(prefix, model_episode)
        torch.save(self.mix_net.state_dict(), save_path_cm)
        return {"policy": save_path_p, "mixer":save_path_m, "central_policy": save_path_cp, "central_mixer":save_path_cm}

    def load_model(self, path, prefix, model_episode: int):
        if model_episode == -1:
            load_path_q = Path(sorted(glob.glob(os.path.join(path, prefix + '_q_*_model')))[-1])
            load_path_mix = Path(sorted(glob.glob(os.path.join(path, prefix + '_mix_*_model')))[-1])
            load_path_cq = Path(sorted(glob.glob(os.path.join(path, prefix + '_cq_*_model')))[-1])
            load_path_cmix = Path(sorted(glob.glob(os.path.join(path, prefix + '_cmix_*_model')))[-1])
        else:
            load_path_q = path / prefix + '_q_{:04}_model'.format(model_episode)
            load_path_mix = path / prefix + '_mix_{:04}_model'.format(model_episode)
            load_path_cq = path / prefix + '_cq_{:04}_model'.format(model_episode)
            load_path_cmix = path / prefix + '_cmix_{:04}_model'.format(model_episode)
        self.q_policy.load_state_dict(torch.load(load_path_q))
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        self.central_q_policy.load_state_dict(torch.load(load_path_cq))
        self.central_q_policy_target.load_state_dict(self.central_q_policy.state_dict())
        self.mix_net.load_state_dict(torch.load(load_path_mix))
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
        self.central_mix_net.load_state_dict(torch.load(load_path_cmix))
        self.central_mix_net_target.load_state_dict(self.central_mix_net.state_dict())

        epoch = int(load_path_q.name.split('_')[2])
        print(f"Trained agent ({epoch}) loaded")
        return epoch + 1

    def decay_exploration(self, episode):
        self.epsilon = max(self.opt.min_epsilon, self.opt.max_epsilon - (self.opt.max_epsilon - self.opt.min_epsilon) * (episode / (0.6 * self.opt.episodes)))

    def take_action(self, observation, hidden=None, dones=None):
        hidden, coordinator_hidden = hidden
        co_q_input = None

        if not self.training:
            actions, hidden, coordinator_hidden = self.q_policy.take_action(observation, hidden, coordinator_hidden, 1-dones, self.opt.min_epsilon)
        else:
            q_out, hidden, coordinator_hidden, inv_q_out, coord_masks, co_q_input, ae_loss = self.q_policy(observation, hidden, coordinator_hidden, 1-dones, eval_coord=True)  # explore both actions and coordination
            actions = self.q_policy.ma_q.sample_action_from_qs(q_out, self.epsilon).squeeze().detach()

            # optimize the coordinator while acting using the max q differences given by a mask respect its inverse
            max_inv_q_a = inv_q_out.max(dim=2)[0]
            max_q_a = q_out.max(dim=2)[0]
            coord_masks = F.softmax(coord_masks.transpose(2, 0), dim=-1)
            w = self.mix_net_target.eval_states(observation).detach()
            if self.opt.mixer != 0:
                inv_pred_q_s = torch.mean(max_inv_q_a.unsqueeze(-1).detach() * w, -1)
                pred_q_s = torch.mean(max_q_a.unsqueeze(-1).detach() * w, -1)
            else:
                inv_pred_q_s = max_inv_q_a.detach() * w
                pred_q_s = max_q_a.detach() * w
            loss_coordinator = torch.sum(torch.sum(torch.sum(
                nn.ReLU()(inv_pred_q_s - pred_q_s).unsqueeze(-1).unsqueeze(-1).expand(coord_masks.shape) * coord_masks, -1), -1) / self.n_agents)

            self.optimizer_coordinator.zero_grad()
            loss_coordinator.backward()
            self.optimizer_coordinator.step()
            self.coord_loss += loss_coordinator
            self.coord_stats.update({"no": (coord_masks[:, :, :, 0]>coord_masks[:, :, :, 1]).sum().item(), "yes": (coord_masks[:, :, :, 0]<coord_masks[:, :, :, 1]).sum().item(), "good": ((inv_pred_q_s-pred_q_s)<0).sum().item(), "bad": ((inv_pred_q_s-pred_q_s)>0).sum().item(), "agree": (inv_q_out.argmax(dim=2) == q_out.argmax(dim=2)).sum().item()})

            # optimize the ae while acting
            if self.q_policy.shared_comm_ae:
                self.optimizer_ae.zero_grad()
                ae_loss.backward()
                self.optimizer_ae.step()
                self.ae_loss += ae_loss

        return actions.detach(), (hidden.detach(), coordinator_hidden.detach()), co_q_input

    def step(self, state, add_in, action, reward, next_state, done):
        self.memory.put((state, action, np.array(reward).tolist(), next_state, np.array(done, dtype=int).tolist()), add_in)
        # drain ea loss got from action in order for logging
        losses = {"ae_loss": self.ae_loss.item()} if self.ae_loss != 0 else {}
        losses.update({"coord_loss": self.coord_loss.item()})
        losses.update(self.coord_stats)
        self.ae_loss = self.coord_loss = 0

        if self.memory.size() > self.warm_up_steps:
            q_loss_epoch = 0

            _chunk_size = self.chunk_size
            for _ in range(self.K_epochs):
                states, comm, coord_masks, actions, rewards, next_states, done_masks = self.memory.sample_chunk(self.batch_size, _chunk_size+1)

                hidden, coord_hidden = self.q_policy.init_hidden(self.batch_size)
                target_hidden, target_coord_hidden = self.q_policy_target.init_hidden(self.batch_size)

                loss_q = 0
                for step_i in range(_chunk_size+1):
                    with torch.no_grad() if step_i == 0 else nullcontext():
                        # reset recurrent info if the chunk contains a restart (all dones)
                        all_done = torch.all((1-done_masks[:, step_i]).bool(),-1)
                        hidden[all_done], coord_hidden[:,:,all_done] = self.q_policy.init_hidden(len(hidden[all_done]))
                        target_hidden[all_done], target_coord_hidden[:,:,all_done] = self.q_policy_target.init_hidden(len(target_hidden[all_done]))
                        # done_masks[:, step_i][all_done] = 1

                        # Pt.1 (net1, net2tgt)
                        # use q_policy to get current state estimatees (act as in past)
                        # then gather values with the actions used
                        # you get chosen_action_qvals
                        # use q_policy_target to get next state estimatees (act at your best)
                        # (double q) get the max action index from live q and use it in target estimates(or do usual max target)
                        # nothing used here?
                        # Pt.2 (net3, net4tgt)
                        # use central_q_policy to get current state estimatees (act as in past)
                        # then gather values with the actions used
                        # you get central_chosen_action_qvals_agents
                        # use central_q_policy_target to get next state estimatees (act at your best)
                        # (double q) get the max action index from live q and use it in target estimates(or do usual max target)
                        # you get central_chosen_action_qvals
                        # Pt.3 (train q_policy,mixer)
                        # a) Use mixer with current states and chosen_action_qvals
                        # b) Use target_central_mixer with current states and central_chosen_action_qvals
                        # q_loss = ws * (a - (rewards + b))**2.sum/sum

                        # Pt.4 (train central_q_policy,central_mixer)
                        # c) Use central_mixer with current states and central_chosen_action_qvals_agents
                        # central_q_loss = (c - (rewards + b))**2.sum/sum

                        # # Calculate estimated Q-Values
                        # mac_out = []
                        # self.mac.init_hidden(batch.batch_size)
                        # for t in range(batch.max_seq_length):
                        #     agent_outs = self.mac.forward(batch, t=t)
                        #     mac_out.append(agent_outs)
                        # mac_out = th.stack(mac_out, dim=1)  # Concat over time
                        #
                        # # Pick the Q-Values for the actions taken by each agent
                        # chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
                        # chosen_action_qvals = chosen_action_qvals_agents
                        #
                        # # Calculate the Q-Values necessary for the target
                        # target_mac_out = []
                        # self.target_mac.init_hidden(batch.batch_size)
                        # for t in range(batch.max_seq_length):
                        #     target_agent_outs = self.target_mac.forward(batch, t=t)
                        #     target_mac_out.append(target_agent_outs)
                        #
                        # # We don't need the first timesteps Q-Value estimate for calculating targets
                        # target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
                        #
                        # # Mask out unavailable actions
                        # target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
                        #
                        # # Max over target Q-Values: Get actions that maximise live Q (for double q-learning)
                        # mac_out_detach = mac_out.clone().detach()
                        # mac_out_detach[avail_actions == 0] = -9999999
                        # cur_max_action_targets, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)
                        # target_max_agent_qvals = th.gather(target_mac_out[:, :], 3, cur_max_actions[:, :]).squeeze(3)
                        #
                        # # Central MAC stuff
                        # central_mac_out = []
                        # self.central_mac.init_hidden(batch.batch_size)
                        # for t in range(batch.max_seq_length):
                        #     agent_outs = self.central_mac.forward(batch, t=t)
                        #     central_mac_out.append(agent_outs)
                        # central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
                        # central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)  # Remove the last dim
                        #
                        # central_target_mac_out = []
                        # self.target_central_mac.init_hidden(batch.batch_size)
                        # for t in range(batch.max_seq_length):
                        #     target_agent_outs = self.target_central_mac.forward(batch, t=t)
                        #     central_target_mac_out.append(target_agent_outs)
                        # central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
                        # # Mask out unavailable actions
                        # central_target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
                        # # Use the Qmix max actions
                        # central_target_max_agent_qvals = th.gather(central_target_mac_out[:, :], 3, cur_max_actions[:, :].unsqueeze(4).repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)
                        # # ---

                        # Mix
                        # chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                        # target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals[:, 1:], batch["state"][:, 1:])
                        # 
                        # # Calculate 1-step Q-Learning targets
                        # targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
                        # 
                        # # Td-error
                        # td_error = (chosen_action_qvals - (targets.detach()))
                        # 
                        # mask = mask.expand_as(td_error)
                        # 
                        # # 0-out the targets that came from padded data
                        # masked_td_error = td_error * mask
                        # 
                        # # Training central Q
                        # central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1])
                        # central_td_error = (central_chosen_action_qvals - targets.detach())
                        # central_mask = mask.expand_as(central_td_error)
                        # central_masked_td_error = central_td_error * central_mask
                        # central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

                        # QMIX loss with weighting
                        # ws = th.ones_like(td_error) * self.args.w
                        # if self.args.hysteretic_qmix:  # OW-QMIX
                        #     ws = th.where(td_error < 0, th.ones_like(td_error) * 1, ws)  # Target is greater than current max
                        # 
                        # else:  # CW-QMIX
                        #     is_max_action = (actions == cur_max_actions[:, :-1]).min(dim=2)[0]
                        #     max_action_qtot = self.target_central_mixer(central_target_max_agent_qvals[:, :-1], batch["state"][:, :-1])
                        #     qtot_larger = targets > max_action_qtot
                        #     ws = th.where(is_max_action | qtot_larger, th.ones_like(td_error) * 1, ws)  # Target is greater than current max
                        # 
                        # qmix_loss = (ws.detach() * (masked_td_error ** 2)).sum() / mask.sum()
                        # 
                        # loss_q += self.lambda_qmix_loss * qmix_loss + self.lambda_central_loss * central_loss

                self.optimizer_policy.zero_grad()
                loss_q.backward()
                if self.grad_clip_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.policy_net_params, self.grad_clip_norm)
                self.optimizer_policy.step()

                # self.logger.log_stat("loss", loss.item(), t_env)
                # self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
                # self.logger.log_stat("grad_norm", grad_norm, t_env)
                # self.logger.log_stat("mixer_norm", mixer_norm, t_env)
                # self.logger.log_stat("agent_norm", agent_norm, t_env)
                # mask_elems = mask.sum().item()
                # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
                # self.logger.log_stat("q_taken_mean",
                #                      (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                # self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                # self.logger.log_stat("central_loss", central_loss.item(), t_env)
                # self.logger.log_stat("w_to_use", w_to_use, t_env)
                q_loss_epoch += loss_q.item()

                num_updates = self.K_epochs
                q_loss_epoch /= num_updates
                losses.update({"q_loss": q_loss_epoch})

            return losses

    def update_target_net(self):
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
        self.central_q_policy_target.load_state_dict(self.central_q_policy.state_dict())
        self.central_mix_target.load_state_dict(self.central_mix.state_dict())

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
