import glob
import os

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from gym import Env
from path import Path
from core.ma_gym.comix.comix_modules import *
from core.ma_gym.replay_buffer import ReplayBuffer
from utils.utils import print_network, get_scheduler, mkdirs


class QMixer(nn.Module):
    def __init__(self, observation_space, hidden_size=32, hypernet_layers=0):
        super(QMixer, self).__init__()
        self.n_agents = len(observation_space)
        self.state_size = sum(np.prod(_.shape) for _ in observation_space.values())
        self.hidden_size = hidden_size

        if hypernet_layers == 0:
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.n_agents))
        elif hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_size, self.hidden_size * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_size, self.hidden_size)
        elif hypernet_layers == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_size, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.hidden_size * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_size, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.hidden_size))
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                               nn.ReLU(),
                               nn.Linear(self.hidden_size, 1))

    def forward_original(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(bs, self.state_size)
        agent_qs = agent_qs.view(bs, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(bs, self.n_agents, self.hidden_size)
        b1 = b1.view(bs, 1, self.hidden_size)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.hidden_size, 1)
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(-1, 1)

        return q_tot

    def forward(self, agent_qs, states):
        states = states.reshape(-1, self.state_size)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # Only second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.n_agents, 1)
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(agent_qs, w_final) + v
        # Reshape and return
        q_tot = y.view(-1, 1)

        return q_tot

    def eval_states(self, states):
        states = states.reshape(-1, self.state_size)
        # Scaling weights used in bmm
        w = torch.abs(self.hyper_w_final(states))

        return w


class Policy(nn.Module):
    """
    Directly predict n values to sample from a learnt distribution: Categorical if Discrete action space else Gaussian
    The output size is dependent to the action size of the environment

    It evaluates in every call a single timestep of the environment, therefore for sequences, multiple calls have to be
    executed.
    """
    def __init__(self, agents_ids, observation_space, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.agents_ids = agents_ids
        self.obs_size = np.prod(list(observation_space.values())[0].shape)
        self.action_space = action_space
        self.action_dtype = torch.int if list(action_space.values())[0].__class__.__name__ == 'Discrete' else torch.float32
        self.hidden_size = base_kwargs["hidden_size"]  # 64
        self.coord_recurrent_size = base_kwargs["coord_recurrent_size"]
        self.shared_comm_ae = False
        self.cnn_input_proc = False
        _to_ma = lambda m: nn.ModuleList([m for _ in range(self.num_agents)])
        # handle implementation with or without vocabulary of measurements

        # preprocess input
        if self.cnn_input_proc:
            self.input_processor = CNNProc(self.obs_size, self.hidden_size)
        else:
            self.input_processor = FCProc(self.obs_size, self.hidden_size)
        # to handle the communication messages
        if self.shared_comm_ae:
            self.comm_size = base_kwargs["ae_comm_size"]  # 10
            self.ae = EncoderDecoder(self.hidden_size+list(self.action_space.values())[0].n, self.comm_size)  # this broke action diversity for agents
        else:
            self.comm_size = self.obs_size+list(self.action_space.values())[0].n
            # in theory the msg is splitted from action processed with`input_processor` (up to `hidden_size`) and concat again with actions
            # in practice the message sent is already the final processed_input+actions for sake of computational effort

        self.ma_q = QNet(self.agents_ids, self.action_space, self.hidden_size, self.action_dtype)  # 64 + maybe also my encoding? in that case add encoding

        self.ma_coordinator = Coordinator(self.agents_ids, self.action_space, plan_size=self.hidden_size+list(self.action_space.values())[0].n, recurrent_size=self.coord_recurrent_size)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            if module_prefix.split('.')[0] == "ae":
                continue
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def to(self, device):
        self._device = device
        super().to(device)
        # the recursion set only the modules parameters not the `_device` variable
        self.ma_q.to(device)
        self.ma_coordinator.to(device)
        return self

    def init_hidden(self, batch_size=1):
        return self.ma_q.init_hidden(batch_size), self.ma_coordinator.init_hidden(batch_size)

    def init_comm_msgs(self, batch_size=1):
        if self.shared_comm_ae:
            return torch.zeros((batch_size, self.num_agents, self.hidden_size+list(self.action_space.values())[0].n
                if self.training else self.comm_size)).to(self._device)
        else:
            return torch.zeros((batch_size, self.num_agents, self.hidden_size+list(self.action_space.values())[0].n)).to(self._device)

    def forward(self, input, rnn_hxs, glob_rnn_hxs, comm, dones, eval_coord=False):
        ae_loss = 0
        # mask hidden
        bs = input.size(0)
        dones = dones.unsqueeze(-1)
        input = input.reshape(bs, self.num_agents, -1)
        # input = input * dones
        # rnn_hxs = rnn_hxs * dones
        # glob_rnn_hxs = glob_rnn_hxs * dones
        # comm = comm * dones
        # shared
        input = input.reshape(bs * self.num_agents, -1)
        x = self.input_processor(input)
        x = x.reshape(bs, self.num_agents, -1)

        qs, rnn_hxs = self.ma_q(x, rnn_hxs, dones)
        solo_actions = self.ma_q.sample_action_from_qs(qs, epsilon=0)  # greedy action selection
        solo_actions_one_hot = torch.zeros_like(qs).scatter_(-1, solo_actions.type(torch.int64).unsqueeze(-1), 1.)

        # create self comm_msg: should encode both processed obs and actions?
        if self.shared_comm_ae:
            current_plans = []
            plans = torch.cat([x, solo_actions_one_hot], dim=-1)  # * dones
            for i in range(self.num_agents):
                enc_x, dec_x = self.ae(plans[:,i])
                ae_loss += F.mse_loss(dec_x, plans[:,i])
                current_plans.append((dec_x if self.training else enc_x).unsqueeze(1))  # (*)
            current_plans = torch.cat(current_plans, dim=1).detach()
        else:
            current_plans = torch.cat([x, solo_actions_one_hot], dim=-1).detach()  # (**)

        # process other's comm_msg masking with dones
        proc_comm = []
        if self.shared_comm_ae:
            if not self.training:
                # (*) being off-policy we cannot give a loss on the base of the decoding
                # of past encoded messages so while training the messages will be sent already decoded
                for i in range(self.num_agents):
                    with torch.no_grad():
                        cf = self.ae.decode(comm[:, i])
                    # if not self.ae_pg:
                    #     cf = cf.detach()
                    proc_comm.append(cf * dones)
            else: proc_comm = comm * dones
        else:
            # (**) pass the entire observation (already processed to not explode the computation time)
            # this is also possible since the shared input processor relaxation
            proc_comm = comm * dones

        # modify your prediction introducing coordination
        qs, inv_q, glob_rnn_hxs, coordinators_mask = self.ma_coordinator(qs, current_plans, proc_comm, rnn_hxs, glob_rnn_hxs, dones, eval_coord)  # TODO rnn_hxs.detach() maybe or maybe not: probably not so grad flows

        return qs, rnn_hxs, glob_rnn_hxs, current_plans, inv_q, coordinators_mask, ae_loss

    def take_action(self, input, rnn_hxs, glob_rnn_hxs, comm, dones, epsilon):
        q, rnn_hxs, glob_rnn_hxs, current_plans, _, _, ae_loss = self.forward(input, rnn_hxs, glob_rnn_hxs, comm, dones)
        # sample action to use in the env
        actions = self.ma_q.sample_action_from_qs(q, epsilon).squeeze()
        return actions.detach(), rnn_hxs.detach(), glob_rnn_hxs.detach(), current_plans.detach(), ae_loss


"""Gym agent"""
class CoordQMixGym(object):

    def __init__(self, opt, env: Env, device: torch.device):
        self.opt = opt
        self.env = env
        self.device = device
        self.train_mode = opt.isTrain
        self.backup_dir = Path(os.path.join(opt.save_path, opt.name))
        self.model_params = {"hidden_size": 64, "coord_recurrent_size": 256, "ae_comm_size": 10}
        mkdirs(self.backup_dir)
        self.start_epoch = 0

        # Setup multi agent modules (dimension with n agents)
        self.q_policy = Policy(env.agents_ids, env.observation_space, env.action_space, base_kwargs=self.model_params).to(device)
        self.q_policy_target = Policy(env.agents_ids, env.observation_space, env.action_space, base_kwargs=self.model_params).to(device)
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        self.mix_net = QMixer(env.observation_space).to(device)
        self.mix_net_target = QMixer(env.observation_space).to(device)
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
            self.q_policy.train()
            self.mix_net.train()
            # initialize optimizers
            self.schedulers, self.optimizers = [], []
            policy_net_params = []
            policy_net_params += self.q_policy.input_processor.parameters()
            policy_net_params += self.q_policy.ma_q.parameters()  # ae parameters were excluded
            policy_net_params += self.q_policy.ma_coordinator.coord_net.parameters()
            policy_net_params += self.q_policy.ma_coordinator.q.parameters()
            policy_net_params += self.mix_net.parameters()

            coordinator_net_params = []
            coordinator_net_params += self.q_policy.ma_coordinator.global_coord_net.parameters()
            coordinator_net_params += self.q_policy.ma_coordinator.boolean_coordinator.parameters()

            self.optimizer_policy = optim.Adam(policy_net_params, lr=opt.lr_q)
            self.optimizer_coordinator = optim.Adam(coordinator_net_params, lr=opt.lr_co)
            self.optimizers.append(self.optimizer_policy)
            self.optimizers.append(self.optimizer_coordinator)
            if self.q_policy.shared_comm_ae:
                self.optimizer_ae = optim.Adam(self.q_policy.ae.parameters(), lr=opt.lr)
                self.optimizers.append(self.optimizer_ae)

            for optimizer in self.optimizers:
                if self.start_epoch > 0: optimizer.param_groups[0].update({"initial_lr": opt.lr})
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
        self.temperature = 1
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
        self.train_mode = mode == "train"
        self.q_policy.train() if self.train_mode else self.q.eval()

    def save_model(self, prefix: str = "", model_episode: int = -1):
        """Save the model"""
        save_path = self.backup_dir / "{}_q_{:04}_model".format(prefix, model_episode)
        torch.save(self.q_policy.state_dict(), save_path)
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
        self.q_policy.load_state_dict(torch.load(load_path_q))
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        self.mix_net.load_state_dict(torch.load(load_path_mix))
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
        epoch = int(load_path_q.name.split('_')[2])
        print(f"Trained agent ({epoch}) loaded")
        return epoch + 1

    def decay_exploration(self, episode):
        self.epsilon = max(self.opt.min_epsilon, self.opt.max_epsilon - (self.opt.max_epsilon - self.opt.min_epsilon) * (episode / (0.6 * self.opt.episodes)))

    def take_action(self, observation, hidden=None, coordinator_hidden=None, comm=None, dones=None, explore=True):
        actions, hidden, coordinator_hidden, comm, ae_loss = self.q_policy.take_action(observation, hidden, coordinator_hidden, comm, dones, self.epsilon if explore else 0)
        # optimize the ae while acting
        if self.q_policy.shared_comm_ae:
            self.optimizer_ae.zero_grad()
            ae_loss.backward()
            self.optimizer_ae.step()
            self.ae_loss += ae_loss

        return actions, hidden, coordinator_hidden, comm

    def step(self, state, action, reward, next_state, done):

        self.memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
        # drain ea loss got from action in order for logging
        losses = {"ae_loss": self.ae_loss.item()} if self.ae_loss != 0 else {}
        self.ae_loss = 0

        if self.memory.size() > self.warm_up_steps:
            q_loss_epoch = 0
            coord_loss_epoch = 0

            # TODO io vorrei clippare i gradient sulla prima action selection quando avviene la coordinazione. probably better not because usually always someone is collaborating and there is less improvement underneath

            _chunk_size = self.chunk_size
            for _ in range(self.K_epochs):
                states, actions, rewards, next_states, done_masks = self.memory.sample_chunk(self.batch_size, _chunk_size)

                hidden, coord_hidden = self.q_policy.init_hidden(self.batch_size)
                comm = self.q_policy.init_comm_msgs(self.batch_size)
                target_hidden, target_coord_hidden = self.q_policy_target.init_hidden(self.batch_size)
                next_comm = self.q_policy_target.init_comm_msgs(self.batch_size)

                loss_q = 0
                loss_coordinator = 0
                for step_i in range(_chunk_size):
                    done_mask = torch.any(1-done_masks[:, step_i],-1)
                    hidden[done_mask], coord_hidden[:,:,done_mask] = self.q_policy.init_hidden(len(hidden[done_mask]))
                    comm[done_mask] = self.q_policy.init_comm_msgs(len(comm[done_mask]))
                    target_hidden[done_mask], target_coord_hidden[:,:,done_mask] = self.q_policy_target.init_hidden(len(target_hidden[done_mask]))
                    next_comm[done_mask] = self.q_policy_target.init_comm_msgs(len(next_comm[done_mask]))

                    q_out, hidden, coord_hidden, comm, inv_q_out, coord_masks, _ = self.q_policy(states[:, step_i], hidden, coord_hidden, comm, done_masks[:,step_i], eval_coord=True)
                    q_a = q_out.gather(2, actions[:, step_i].unsqueeze(-1).long()).squeeze(-1)
                    max_inv_q_a = inv_q_out.max(dim=2)[0].squeeze(-1)
                    pred_q = self.mix_net(q_a, states[:, step_i])  # TODO mask states and qa for dones?

                    q_target, target_hidden, target_coord_hidden, next_comm, _, _, _ = self.q_policy_target(next_states[:, step_i], target_hidden.detach(), target_coord_hidden.detach(), next_comm, done_masks[:,1+step_i])
                    max_q_target = q_target.max(dim=2)[0].squeeze(-1)
                    next_q_total = self.mix_net_target(max_q_target, next_states[:, step_i])
                    target = (rewards[:, step_i] * done_masks[:,1+step_i]).sum(dim=1, keepdims=True) + (self.gamma * next_q_total)
                    loss_q += F.smooth_l1_loss(pred_q, target.detach())

                    # train the coordinator with the q differences 
                    coord_masks = F.gumbel_softmax(coord_masks.transpose(2, 0), tau=0.01, dim=-1)
                    w = self.mix_net_target.eval_states(states[:, step_i]).detach()
                    inv_pred_q_s = (max_inv_q_a.detach() * w).unsqueeze(-1).expand(coord_masks.shape[:3])
                    pred_q_s = (q_a.detach() * w).unsqueeze(-1).expand(coord_masks.shape[:3])
                    stacked_q_s = torch.stack([inv_pred_q_s, pred_q_s], dim=-1)
                    loss_coordinator += torch.sum(nn.ReLU()(torch.sum(stacked_q_s * (1 - coord_masks), -1) - torch.sum(stacked_q_s * coord_masks, -1)))

                self.optimizer_coordinator.zero_grad()
                loss_coordinator.backward()
                self.optimizer_coordinator.step()

                self.optimizer_policy.zero_grad()
                loss_q.backward()
                self.optimizer_policy.step()
                # torch.nn.utils.clip_grad_norm_(self.q_policy.parameters(), self.grad_clip_norm, norm_type=2)
                # torch.nn.utils.clip_grad_norm_(self.mix_net.parameters(), self.grad_clip_norm, norm_type=2)

                q_loss_epoch += loss_q.item()
                coord_loss_epoch += loss_coordinator.item()

            num_updates = self.K_epochs * _chunk_size
            q_loss_epoch /= num_updates
            coord_loss_epoch /= num_updates
            losses.update({"q_loss": q_loss_epoch, "coord_loss": coord_loss_epoch})

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
