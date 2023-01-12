import glob
import os
from contextlib import nullcontext

import torch.optim as optim
from gym import Env
from path import Path

from core.base_agent import BaseAgent
from core.ma_gym.comix.comix_net import *
from core.ma_gym.replay_buffer import ReplayBuffer
from utils.utils import print_network, get_scheduler, mkdirs


"""Gym agent"""
class CoordQMixGymAgent(BaseAgent):

    def __init__(self, opt, env: Env, device: torch.device):
        super().__init__(opt, env, device)
        self.model_params = {"eval_coord_mask": opt.coord_mask_type, "ae_comm": bool(opt.ae_comm), "hidden_size": 32, "coord_recurrent_size": 64, "ae_comm_size": 16, "input_proc_size": 1, "cnn_input_proc": bool(opt.cnn_input_proc)}
        self.mixer_params = {"hidden_size": 32}

        # Setup multi agent modules (dimension with n agents)
        self.q_policy = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.model_params).to(device)
        self.q_policy_target = QPolicy(env.agents_ids, env.observation_space, env.action_space, model_params=self.model_params).to(device)
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        # no need of having a target for mask predictor module
        self.q_policy_target.ma_coordinator.global_coord_net = self.q_policy.ma_coordinator.global_coord_net
        self.q_policy_target.ma_coordinator.boolean_coordinator = self.q_policy.ma_coordinator.boolean_coordinator
        self.mix_net = QMixer(env.observation_space, hidden_size=self.mixer_params["hidden_size"], hypernet_layers=opt.mixer).to(device)
        self.mix_net_target = QMixer(env.observation_space, hidden_size=self.mixer_params["hidden_size"], hypernet_layers=opt.mixer).to(device)
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
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
        if opt.isTrain:
            self.q_policy.train()
            self.mix_net.train()
            # initialize optimizers
            self.schedulers, self.optimizers, self.initial_lr = [], [], []
            policy_net_params = []
            policy_net_params += self.q_policy.get_policy_parameters()
            policy_net_params += self.mix_net.parameters()

            self.optimizer_policy = optim.Adam(policy_net_params, lr=opt.lr_q)
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
        self.grad_clip_norm = opt.grad_clip_norm  # 5
        self.lambda_coord = 0.1
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

    def decay_exploration(self, episode):
        self.epsilon = max(self.opt.min_epsilon, self.opt.max_epsilon - (self.opt.max_epsilon - self.opt.min_epsilon) * (episode / (0.6 * self.opt.episodes)))

    def take_action(self, observation, hidden=None, dones=None):
        hidden, coordinator_hidden = hidden
        co_q_input = None

        if not self.training:
            actions, hidden, coordinator_hidden = self.q_policy.take_action(observation, hidden, coordinator_hidden, 1-dones, self.opt.min_epsilon)
        else:
            # Act then do on-policy optimization for Coordinator and Autoencoder
            q_out, hidden, coordinator_hidden, inv_q_out, coord_masks, co_q_input, mid_losses = self.q_policy(observation, hidden, coordinator_hidden, 1-dones, eval_coord=True)  # explore both actions and coordination
            actions = self.q_policy.ma_q.sample_action_from_qs(q_out, self.epsilon).squeeze().detach()

            distance_loss, ae_loss = mid_losses
            # optimize the coordinator while acting using the max q differences given by a mask respect its inverse
            loss_coordinator = distance_loss
            max_inv_q_a = inv_q_out.max(dim=-1)[0]
            max_q_a = q_out.max(dim=-1)[0]
            coord_masks = coord_masks.transpose(2, 0)  # F.softmax(coord_masks.transpose(2, 0), dim=-1)
            w = self.mix_net_target.eval_states(observation).detach()
            if self.q_policy.eval_coord_mask == "optout":
                if self.opt.mixer == 0: raise Exception("The dev messed up")
                inv_pred_q_s = torch.mean(max_inv_q_a.unsqueeze(-1).detach() * w.unsqueeze(2), -1)
                pred_q_s = torch.mean(max_q_a.unsqueeze(-1).detach() * w, -1)
                loss_coordinator += torch.sum(torch.sum(
                    nn.ReLU()(inv_pred_q_s - pred_q_s.unsqueeze(-1).expand(max_inv_q_a.shape)).unsqueeze(-1).expand(coord_masks.shape) * coord_masks, -1))
            else:
                if self.opt.mixer != 0:
                    inv_pred_q_s = torch.mean(max_inv_q_a.unsqueeze(-1).detach() * w, -1)
                    pred_q_s = torch.mean(max_q_a.unsqueeze(-1).detach() * w, -1)
                else:
                    inv_pred_q_s = max_inv_q_a.detach() * w
                    pred_q_s = max_q_a.detach() * w
                loss_coordinator += torch.sum(torch.sum(torch.sum(
                    nn.ReLU()(inv_pred_q_s - pred_q_s).unsqueeze(-1).unsqueeze(-1).expand(coord_masks.shape) * coord_masks, -1), -1) / self.n_agents)


            self.optimizer_coordinator.zero_grad()
            loss_coordinator.backward()
            self.optimizer_coordinator.step()
            self.coord_loss += loss_coordinator
            self.coord_stats.update({"no": (coord_masks[:, :, :, 0]>coord_masks[:, :, :, 1]).sum().item(), "yes": (coord_masks[:, :, :, 0]<coord_masks[:, :, :, 1]).sum().item(), "good": ((inv_pred_q_s-pred_q_s.expand(inv_pred_q_s.shape))<=0).sum().item(), "bad": ((inv_pred_q_s-pred_q_s.expand(inv_pred_q_s.shape))>0).sum().item(), "agree": (inv_q_out.view(q_out.size(0),q_out.size(1),-1).argmax(2)%q_out.size(2) == q_out.argmax(dim=2)).sum().item()})

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

                        # q policy training with respect the target q policy network
                        q_a, hidden, coord_hidden = self.q_policy.eval_action(states[:, step_i], hidden, coord_hidden, done_masks[:, step_i], comm[:, step_i], coord_masks[:, step_i], actions[:, step_i])
                        pred_q = self.mix_net(q_a, states[:, step_i], done_masks[:, step_i])

                        q_target, target_hidden, target_coord_hidden, _, _, _, _ = self.q_policy_target(next_states[:, step_i], target_hidden.detach(), target_coord_hidden.detach(), done_masks[:, step_i+1])
                        max_q_target = q_target.max(dim=2)[0].squeeze(-1)
                        next_q_total = self.mix_net_target(max_q_target, next_states[:, step_i], done_masks[:, step_i+1]).detach()
                        target = (rewards[:, step_i] * done_masks[:, step_i]).sum(dim=1, keepdims=True) + (self.gamma * next_q_total)
                        loss_q += F.l1_loss(pred_q, target) if step_i > 0 else 0

                self.optimizer_policy.zero_grad()
                loss_q.backward()
                if self.grad_clip_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.q_policy.get_policy_parameters(), self.grad_clip_norm, norm_type=2)
                    torch.nn.utils.clip_grad_norm_(self.mix_net.parameters(), self.grad_clip_norm, norm_type=2)
                self.optimizer_policy.step()

                q_loss_epoch += loss_q.item()

            num_updates = self.K_epochs
            q_loss_epoch /= num_updates
            losses.update({"q_loss": q_loss_epoch})

        return losses

    def update_target_net(self):
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
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
