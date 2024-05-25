import glob
import os
from argparse import Namespace

import torch.optim as optim
from gym import Env
from path import Path

from core.base_agent import BaseAgent
from core.ma_gym.memory.roll_storage import RolloutStorage
from core.ma_gym.comix.comix_net import *
from utils.utils import print_network, get_scheduler, mkdirs

DEBUG = True

"""Gym agent for all agents in the environment"""
class CoordQMixGymAgent(BaseAgent):

    def __init__(self, opt: Namespace, env: Env, device: torch.device):
        super().__init__(opt, env, device)
        self.q_params = {"num_dummy_agents": env.n_agents_dummy, "eval_coord_mask": opt.coord_mask_type, "ae_comm": bool(opt.ae_comm), "hidden_size": opt.hi, "coord_recurrent_size": opt.hc, "ae_comm_size": 16, "input_proc_size": opt.hs, "norm_layer": opt.norm_layer, "cnn_input_proc": bool(opt.cnn_input_proc)}
        self.mixer_params = {"hidden_size": opt.hm}

        # Create multi-agent networks (stacked individually for n agents)
        self.q_policy = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.q_params).to(device)
        self.q_policy_target = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.q_params).to(device)
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        # the 'mask predictor module' is trained separately and doesn't need a target network
        self.q_policy_target.ma_coordinator = self.q_policy.ma_coordinator
        self.mix_net = QMixer(env.observation_space, model_params=self.mixer_params).to(device)
        self.mix_net_target = QMixer(env.observation_space, model_params=self.mixer_params).to(device)
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())

        # Load weights
        self.training = opt.isTrain
        self.fine_tuning = opt.fine_tune if self.training else False
        if self.training:
            self.backup_dir = opt.backup_dir
            if opt.continue_train is not None:
                self.start_epoch = self.load_model(self.backup_dir, "policy", opt.continue_train)
        else:
            self.load_model(opt.models_path, "policy", opt.model_epoch)

        # Setup networks train/eval states
        self.q_policy_target.eval()
        self.mix_net_target.eval()
        if self.training:
            self.q_policy.train()
            self.mix_net.train()
            # initialize optimizers
            self.schedulers, self.optimizers, self.initial_lr = [], [], []
            policy_net_params = []
            policy_net_params += self.q_policy.get_policy_parameters()
            policy_net_params += self.mix_net.parameters()
            
            if opt.optimizer == "adam":
                self.optimizer_policy = optim.Adam(policy_net_params, lr=opt.lr_q, weight_decay=opt.lr_weight_decay)
            else:
                self.optimizer_policy = optim.RMSprop(policy_net_params, lr=opt.lr_q, alpha=0.97, eps=1e-8, weight_decay=opt.lr_weight_decay)
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

        if self.fine_tuning:
            for param in []+list(self.mix_net.parameters())+self.q_policy.get_policy_parameters()+self.q_policy.get_coordinator_parameters():
                param.requires_grad = False
            for param in list(self.q_policy.co_q_linear.parameters()):
                param.requires_grad = True
                self.q_policy.delayed_comm = True

        if not self.training and not self.fine_tuning:
            self.q_policy.eval()

        print_network(self.q_policy)
        print_network(self.mix_net)

        # train cycle params
        self.train = True
        self.no_op = env.no_op
        if self.training or self.fine_tuning:
            self.memory = RolloutStorage(opt.max_buffer_len, env.n_agents, env.n_agents_dummy, env.observation_space, env.action_space, self.q_policy.plan_size).to(device)
            self.K_epochs = opt.K_epochs
            self.coord_K_epochs = opt.coord_K_epochs
            self.batch_size = opt.batch_size
            self.warm_up_steps = opt.min_buffer_len
            self.gamma = opt.gamma
            self.chunk_size = opt.chunk_size
            self.grad_clip_norm = opt.grad_clip_norm
            self.lambda_coord = opt.lambda_coord
            self.lambda_q = opt.lambda_q/self.chunk_size
            self.tau = opt.tau
            self.coord_loss = torch.zeros(1, device=device)
            self.accumulate_loss_coordinator = torch.zeros(1, device=device)
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
        """Save the model"""
        assert self.training or self.fine_tuning, "Calling `save_model` method is not allowed in evaluation mode."
        save_path_p = self.backup_dir / "{}_q_{:04}_model".format(prefix, model_episode)
        torch.save(self.q_policy.state_dict(), save_path_p)
        save_path_m = self.backup_dir / "{}_mix_{:04}_model".format(prefix, model_episode)
        torch.save(self.mix_net.state_dict(), save_path_m)
        return {"policy": save_path_p, "mixer":save_path_m}

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
        self.q_policy.load_state_dict(torch.load(load_path_q))
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        self.mix_net.load_state_dict(torch.load(load_path_mix))
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())

        epoch = int(load_path_q.name.split('_')[2])
        print(f"Trained agent ({epoch}) loaded")
        return epoch + 1

    def _soft_update(self):
        # for param_target, param in zip(self.q_policy_target.get_policy_parameters(), self.q_policy.get_policy_parameters()):
        #     param_target.data.copy_((param_target.data * (1.0 - self.tau)) + (param.data * self.tau))
        # for param_target, param in zip(self.mix_net_target.parameters(), self.mix_net.parameters()):
        #     param_target.data.copy_((param_target.data * (1.0 - self.tau)) + (param.data * self.tau))
        pass

    def take_action(self, observation, hidden, dones, prev_intents=None, comm_delays=None):
        observation = observation.unsqueeze(0)
        hidden, coordinator_hidden = hidden
        dones_mask = (1-dones).unsqueeze(0)
        if comm_delays is not None: comm_delays = comm_delays.unsqueeze(0)

        if not self.training:
            actions, hidden, coordinator_hidden, additional_input = self.q_policy.take_action(observation, hidden, coordinator_hidden, dones_mask, prev_intents, comm_delays, temperature=0.1)
        else:
            # Act, then do on-policy optimization for Coordinator and Autoencoder
            coord_training = not self.fine_tuning and self.memory.size() >= self.warm_up_steps
            ae_training = not self.fine_tuning and self.q_policy.shared_comm_ae

            q_out, hidden, coordinator_hidden, inv_q_out, coord_masks, additional_input, ae_loss = self.q_policy(observation, hidden, coordinator_hidden, dones_mask, train_coord=coord_training, train_ae=ae_training)  # explore both actions and coordination
            actions = self.q_policy.sample_action_from_qs(q_out, temperature=1)

            # Optimize the coordination module while acting (avoid while Q is still dumb)
            if coord_training:
                # optimize the coordinator while acting using the max q differences given by a mask respect its inverse
                dones_mask = dones_mask.squeeze(-1)
                max_q_a = q_out.max(dim=-1)[0] * dones_mask
                coord_masks = coord_masks.transpose(2, 0)
                w = self.mix_net_target.eval_states(observation).detach()
                if self.q_policy.eval_coord_mask == "optout":
                    max_inv_q_a = inv_q_out.max(dim=-1)[0] * dones_mask.unsqueeze(-1) * dones_mask.unsqueeze(-2)
                    inv_pred_q_s = torch.mean(max_inv_q_a.unsqueeze(-1).detach() * w.unsqueeze(2), -1)
                    pred_q_s = torch.mean(max_q_a.unsqueeze(-1).detach() * w, -1)
                    inv_pred_q_s = torch.cat([inv_pred_q_s, pred_q_s.unsqueeze(1).expand((max_inv_q_a.size(0), self.q_policy.num_agents_dummy-self.q_policy.num_agents, self.q_policy.num_agents))], dim=1)
                    loss = torch.sum(
                        nn.ReLU()(inv_pred_q_s - pred_q_s.unsqueeze(1).expand((max_inv_q_a.size(0), self.q_policy.num_agents_dummy, self.q_policy.num_agents))) * coord_masks.max(-1)[0])
                else:
                    max_inv_q_a = inv_q_out.max(dim=-1)[0] * dones_mask
                    inv_pred_q_s = torch.mean(max_inv_q_a.unsqueeze(-1).detach() * w, -1)
                    pred_q_s = torch.mean(max_q_a.unsqueeze(-1).detach() * w, -1)
                    loss = torch.sum(torch.sum(torch.sum(
                        nn.ReLU()(inv_pred_q_s - pred_q_s).unsqueeze(-1).unsqueeze(-1).expand(coord_masks.shape) * coord_masks, -1), -1) / self.n_agents)

                if self.train and loss > 0:
                    self.accumulate_loss_coordinator += loss
                    self.coord_updates += self.coord_K_epochs

                    if self.coord_updates >= 1:
                        self.optimizer_coordinator.zero_grad()
                        self.accumulate_loss_coordinator.backward()
                        self.optimizer_coordinator.step()
                        self.coord_loss += self.accumulate_loss_coordinator
                        diag = (~torch.eye(self.q_policy.num_agents, dtype=bool)).int().unsqueeze(0)
                        # communication choice resulted in best q, normalized 
                        good_ratio = ((((inv_pred_q_s-pred_q_s.expand(inv_pred_q_s.shape))<=0)[:,:self.q_policy.num_agents]*diag).sum()/self.q_policy.num_agents/(self.q_policy.num_agents-1)).item()
                        # messages accepted normalized 
                        yes = (((coord_masks[:, :, :, 0]<coord_masks[:, :, :, 1])[:,:self.q_policy.num_agents]*diag).sum()/self.q_policy.num_agents/(self.q_policy.num_agents-1)).item()
                        no = (((coord_masks[:, :, :, 0]>coord_masks[:, :, :, 1])[:,:self.q_policy.num_agents]*diag).sum()/self.q_policy.num_agents/(self.q_policy.num_agents-1)).item()
                        # action correct independently by the communication normalized
                        agree = ((inv_q_out.view(q_out.size(0),q_out.size(1),-1).argmax(2)%q_out.size(2) == q_out.argmax(dim=2)).sum()/self.q_policy.num_agents).item()
                        self.coord_stats.update({ "yes": yes, "no": no, "good": good_ratio, "agree": agree})
                        if DEBUG:
                            self.coord_stats.update({p[0]: p[1].grad.norm(2).item() for p in self.q_policy.ma_coordinator.coord_net_modules.named_parameters() if "weight" in p[0] and p[1].grad is not None})

                        self.coord_updates -= 1
                        self.accumulate_loss_coordinator = torch.zeros(1, device=self.accumulate_loss_coordinator.device)

            # optimize the ae while acting
            if ae_training:
                self.optimizer_ae.zero_grad()
                ae_loss.backward()
                self.optimizer_ae.step()
                self.ae_loss += ae_loss

        actions = actions.squeeze().unsqueeze(-1)
        actions[dones.squeeze().bool()] = self.no_op
        return actions.detach(), (hidden.detach(), coordinator_hidden.detach()), additional_input

    def step(self, state, add_in, action, reward, next_state, done):
        self.memory.put(state, *([a.squeeze(0) for a in add_in[:2]] if add_in is not None else [None, None]), action, reward, next_state, done)

        # drain ea loss got from action in order for logging
        losses = {"ae_loss": self.ae_loss.item(), "coord_loss": self.coord_loss.item()}
        losses.update(self.coord_stats)
        self.ae_loss, self.coord_loss = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        num_updates = 0

        if self.train and self.memory.size() >= self.warm_up_steps:
            self.updates += self.K_epochs
            if self.updates >= 1:
                q_loss_epoch = 0
                num_updates = int(self.updates)
                for _ in range(num_updates):
                    states, comm, coord_masks, actions, rewards, next_states, done_masks = self.memory.sample_chunk(self.batch_size, self.chunk_size)

                    hidden, coord_hidden = self.q_policy.init_hidden(self.batch_size)
                    target_hidden, target_coord_hidden = self.q_policy_target.init_hidden(self.batch_size)

                    loss_q = 0
                    for step_i in range(self.chunk_size):
                        # reset recurrent info /*if the chunk contains a restart (all dones)*/ if done
                        reset = (1-done_masks[:, step_i]).squeeze(-1).bool()
                        next_reset = (1-done_masks[:, step_i+1]).squeeze(-1).bool()
                        m, cm, tm, tcm = *self.q_policy.init_hidden(len(reset)), *self.q_policy_target.init_hidden(len(next_reset))
                        hidden[reset], coord_hidden.permute(2,0,1,3)[reset] = m[reset], cm.permute(2,0,1,3)[reset]
                        target_hidden[next_reset], target_coord_hidden.permute(2,0,1,3)[next_reset] = tm[next_reset], tcm.permute(2,0,1,3)[next_reset]
                        done_masks[:, step_i][torch.all(reset,-1)] = 1  # fresh start when needed
                        next_done_mask = done_masks[:, step_i+1].clone()
                        # trick to run the q_policy_target in the new episode, if we are in between the reset, in order to update the target_hidden while the q_target value in putput is masked to 0
                        if step_i+1 < self.chunk_size:
                            next_done_mask[torch.all(next_reset,-1)] = 1  # i.e. step_i+2
                            next_states[:, step_i][torch.all(next_reset,-1)] = states[:, step_i+1][torch.all(next_reset,-1)]

                        # q policy training with respect the target q policy network
                        q_a, hidden, coord_hidden = self.q_policy.eval_action(states[:, step_i], hidden, coord_hidden, done_masks[:, step_i], comm[:, step_i], coord_masks[:, step_i], actions[:, step_i])
                        q_a[~done_masks[:, step_i].squeeze(-1).bool()] = 0
                        pred_q = self.mix_net(q_a, states[:, step_i])

                        q_target, target_hidden, target_coord_hidden, _, _, _, _ = self.q_policy_target(next_states[:, step_i], target_hidden.detach(), target_coord_hidden.detach(), next_done_mask)
                        max_q_target = q_target.max(dim=-1)[0].squeeze(-1)
                        max_q_target = max_q_target * next_done_mask.squeeze(-1)
                        next_q_total = self.mix_net_target(max_q_target, next_states[:, step_i]).detach()
                        next_q_total[torch.all(next_reset,-1)] = 0  # the step+1 is a new episode 

                        target = (rewards[:, step_i] * done_masks[:, step_i]).squeeze(-1).sum(dim=1, keepdims=True) + (self.gamma * next_q_total)
                        loss_q += F.mse_loss(pred_q, target) * self.lambda_q

                    self.optimizer_policy.zero_grad()
                    loss_q.backward()
                    if self.grad_clip_norm != 0:
                        torch.nn.utils.clip_grad_norm_(self.q_policy.get_policy_parameters(), self.grad_clip_norm, norm_type=2)
                        torch.nn.utils.clip_grad_norm_(self.mix_net.parameters(), self.grad_clip_norm, norm_type=2)
                    self.optimizer_policy.step()

                    self._soft_update()

                    q_loss_epoch += loss_q.item()

                q_loss_epoch /= num_updates
                self.updates -= num_updates
                losses.update({"q_loss": q_loss_epoch})
                if DEBUG:
                    losses.update({p[0]: p[1].grad.norm(2).item() for p in self.q_policy.ma_q.gru_modules.named_parameters() if "weight" in p[0] and p[1].grad is not None})

        return num_updates, losses

    def update_target_net(self):
        self.q_policy_target.input_processor.load_state_dict(self.q_policy.input_processor.state_dict())
        self.q_policy_target.ma_q.load_state_dict(self.q_policy.ma_q.state_dict())
        self.q_policy_target.intent_extractor.load_state_dict(self.q_policy.intent_extractor.state_dict())
        self.q_policy_target.co_q_linear.load_state_dict(self.q_policy.co_q_linear.state_dict())
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
        print('Target networks updated')

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
