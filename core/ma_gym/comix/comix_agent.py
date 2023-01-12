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
from core.ma_gym.comix.comix_modules import *
from core.ma_gym.replay_buffer import ReplayBuffer
from utils.utils import print_network, get_scheduler, mkdirs


class QMixer(nn.Module):
    def __init__(self, observation_space, hidden_size=32, hypernet_layers=1):
        super(QMixer, self).__init__()
        self.num_agents = len(observation_space)
        self.state_size = sum(np.prod(_.shape) for _ in observation_space.values())
        self.hidden_size = hidden_size
        self.hypernet_layers = hypernet_layers

        if hypernet_layers == 0:
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.num_agents))
        elif hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_size, self.hidden_size * self.num_agents)
            self.hyper_w_final = nn.Linear(self.state_size, self.hidden_size)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_size, self.hidden_size * 2 * self.num_agents),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size * 2 * self.num_agents, self.hidden_size * self.num_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_size, self.hidden_size * 2),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size * 2, self.hidden_size))
        self.hyper_b_1 = nn.Linear(self.state_size, self.hidden_size)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                               nn.ReLU(),
                               nn.Linear(self.hidden_size, 1))

    def forward(self, agent_qs, states, dones_mask):
        bs = agent_qs.size(0)
        dones_mask = dones_mask.unsqueeze(-1)
        states = states.reshape(bs, self.num_agents, -1) * dones_mask

        if self.hypernet_layers != 0:
            states = states.reshape(bs, self.state_size)
            agent_qs = agent_qs.view(bs, 1, self.num_agents)
            # First layer
            w1 = torch.abs(self.hyper_w_1(states))
            b1 = self.hyper_b_1(states)
            w1 = w1.view(bs, self.num_agents, self.hidden_size)
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
        else:
            states = states.reshape(bs, self.state_size)
            agent_qs = agent_qs.view(bs, 1, self.num_agents)
            # Only second layer
            w_final = torch.abs(self.hyper_w_final(states))
            w_final = w_final.view(-1, self.num_agents, 1)
            v = self.V(states).view(-1, 1, 1)
            # Compute final output
            y = torch.bmm(agent_qs, w_final) + v
            # Reshape and return
            q_tot = y.view(-1, 1)

            return q_tot

    def eval_states(self, states):
        if self.hypernet_layers != 0:
            states = states.reshape(-1, self.state_size)
            # Scaling weights used in bmm
            w1 = torch.abs(self.hyper_w_1(states))

            return w1.reshape(-1, self.num_agents, self.hidden_size)
        else:
            states = states.reshape(-1, self.state_size)
            # Scaling weights used in bmm
            w = torch.abs(self.hyper_w_final(states))

            return w


class QPolicy(nn.Module):
    """
    Directly predict n values to sample from a learnt distribution: Categorical if Discrete action space else Gaussian
    The output size is dependent to the action size of the environment

    It evaluates in every call a single timestep of the environment, therefore for sequences, multiple calls have to be
    executed.
    """
    def __init__(self, agents_ids, observation_space, action_space, model_params=None):
        super(QPolicy, self).__init__()
        if model_params is None:
            model_params = {}
        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.agents_ids = agents_ids
        self.obs_shape = list(observation_space.values())[0].shape
        self.obs_size = np.prod(self.obs_shape)
        self.action_space = action_space
        self.action_dtype = torch.int if list(action_space.values())[0].__class__.__name__ == 'Discrete' else torch.float32
        self.hidden_size = model_params.get("hidden_size", 64)
        self.coord_recurrent_size = model_params.get("coord_recurrent_size", 256)
        self.shared_comm_ae = model_params.get("ae_comm", False)
        self.cnn_input_proc = model_params.get("cnn_input_proc", False)
        self.eval_coord_mask = model_params.get("eval_coord_mask", False)

        self.action_size = list(self.action_space.values())[0].n  # this breaks action diversity for agents
        self.solo_recurrent_size = self.hidden_size
        self.plan_size = self.num_agents+self.hidden_size+self.action_size
        self.ids_one_hot = torch.eye(self.num_agents).to(self._device)
        _to_ma = lambda m, args, kwargs: nn.ModuleList([m(*args, **kwargs) for _ in range(self.num_agents)])
        _init_fc_norm = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0))

        # preprocess input to extract relevant features (shared)
        if self.cnn_input_proc:
            self.input_processor = CNNProc(3, self.hidden_size, size=model_params.get("input_proc_size", 0))
        else:
            self.input_processor = FCProc(self.obs_size, self.hidden_size, size=model_params.get("input_proc_size", 0))
        # handle the communication messages
        if self.shared_comm_ae:
            self.comm_size = model_params.get("ae_comm_size", 16)
            self.ae = EncoderDecoder(self.hidden_size+self.action_size, self.comm_size)
        else:
            # in theory the msg is formed as concatenation of observation and action
            self.comm_size = self.obs_size+self.action_size
            # in practice the message sent is already the final processed_input+actions for sake of computational effort
            self.comm_size = self.hidden_size+self.action_size
        # Q network to take decisions independently of others
        self.ma_q = QNet(self.agents_ids, self.action_space, self.hidden_size, self.action_dtype)  # 64 + maybe also my encoding? in that case add encoding
        # Produce the coordination maks 
        self.ma_coordinator = Coordinator(self.agents_ids, self.action_space, plan_size=self.plan_size, solo_recurrent_size=self.solo_recurrent_size, coord_recurrent_size=self.coord_recurrent_size)
        # Coordinated Q network to `slightly` adjust your decisions: the role of these modules is equivalent to a QNet but are created as it is for granularity of usage
        self.co_q_net = _to_ma(nn.GRUCell, (self.plan_size, self.solo_recurrent_size), {})
        self.co_q_linear = nn.ModuleList([_init_fc_norm(nn.Linear(self.solo_recurrent_size, action_space[id].n)) for id in agents_ids])
        for name, param in self.co_q_net.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def get_policy_parameters(self):
        policy_net_params = []
        policy_net_params += self.input_processor.parameters()
        policy_net_params += self.ma_q.parameters()
        policy_net_params += self.co_q_net.parameters()
        policy_net_params += self.co_q_linear.parameters()

        return policy_net_params

    def get_coordinator_parameters(self):
        coordinator_net_params = []
        coordinator_net_params += self.ma_coordinator.global_coord_net.parameters()
        coordinator_net_params += self.ma_coordinator.boolean_coordinator.parameters()

        return coordinator_net_params

    def to(self, device):
        self._device = device
        super().to(device)
        # ...the standard recursion applied to submodules set only the modules parameters and not the `_device` variable
        self.ids_one_hot = self.ids_one_hot.to(device)
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

    def _process_msgs(self, comm):
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
                    proc_comm.append(cf.unsqueeze(1))
                proc_comm = torch.cat(proc_comm, dim=1)
            else:
                proc_comm = comm
        else:
            # (**) pass the entire obs already preprocessed by feature extractor (to not explode the computation time)
            # this is also possible since the shared input processor relaxation
            proc_comm = comm
        return proc_comm

    def _modify_qs(self, action_logits, hiddens, comm_plans, masks):
        # Compute q modifications on the base of the forced coordination induced by the coordination boolean mask
        # `action_logits` (req_grad), `masks` (detached), `comm_msgs` (detached)
        q_values = []
        for i, id in enumerate(self.agents_ids):
            rnn_hxs = hiddens[:, i].clone()
            for j in range(self.num_agents):
                if i == j: continue
                comm_plans_masked = comm_plans[:, j] * masks[i, j]
                batch_mask = torch.any(comm_plans_masked, -1).unsqueeze(-1)
                batch_comm_plans_masked = torch.masked_select(comm_plans_masked, batch_mask).reshape(-1, comm_plans_masked.size(-1))
                batch_rnn_hxs = torch.masked_select(rnn_hxs, batch_mask).reshape(-1, rnn_hxs.size(-1))
                if len(batch_comm_plans_masked) > 0:  # certain versions of PyTorch don't like empty batches
                    batch_rnn_hxs = self.co_q_net[i](batch_comm_plans_masked, batch_rnn_hxs)
                    rnn_hxs = rnn_hxs.masked_scatter(batch_mask, batch_rnn_hxs)
            q_values.append((action_logits[:, i] + self.co_q_linear[i](rnn_hxs)).unsqueeze(1))  # NOTE xke questo funzioni il mio hidden deve essere abbastanza informativo da poter inferire gli action_logits precedentemente emessi e di conseguenza con la q una modifica a questi... se non funziona mettere tutto nella q
        q_values = torch.cat(q_values, dim=1)

        return q_values

    def forward(self, input, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans=None, coord_masks=None, eval_coord=False):
        assert not eval_coord or eval_coord and (comm_plans is None and coord_masks is None), "The arguments combination passed do not match a std wanted behaviour."
        assert not (comm_plans is None and coord_masks is None) or (comm_plans is None and coord_masks is None), "The arguments combination passed is not valid."
        ae_loss = 0
        distance_loss = 0
        # mask hidden
        bs = input.size(0)
        dones_mask = dones_mask.unsqueeze(-1)

        # shared input processing
        if self.cnn_input_proc:
            input = input.reshape(bs * self.num_agents, *self.obs_shape).transpose(-1, -3)
        else:
            input = input.reshape(bs * self.num_agents, -1)
        x = self.input_processor(input)
        x = x.reshape(bs, self.num_agents, -1)

        # --- Solo action ---
        solo_qs, rnn_hxs = self.ma_q(x, rnn_hxs)
        solo_actions = self.ma_q.sample_action_from_qs(solo_qs, epsilon=0.15)  # greedy action selection
        solo_actions_one_hot = torch.zeros_like(solo_qs).scatter_(-1, solo_actions.type(torch.int64).unsqueeze(-1), 1.)

        # --- Communication ---
        # create current comm_msg to send: should encode both processed obs and actions? even if you are done the msg is created
        if self.shared_comm_ae:
            current_plans = []
            plans = torch.cat([self.ids_one_hot.repeat(bs, 1, 1), x, solo_actions_one_hot], dim=-1)
            for i in range(self.num_agents):
                enc_x, dec_x = self.ae(plans[:, i])
                ae_loss += F.mse_loss(dec_x, plans[:, i])
                current_plans.append((dec_x if self.training else enc_x).unsqueeze(1))  # (*)
            current_plans = torch.cat(current_plans, dim=1).detach()
        else:
            current_plans = torch.cat([self.ids_one_hot.repeat(bs, 1, 1), x, solo_actions_one_hot], dim=-1).detach()  # (**)
        # retrieve incoming messages
        comm_msgs = current_plans  # comm_msgs are current timestep plans
        # process incoming comm_msg from others. It needs to be masked with dones
        proc_comm = (self._process_msgs(comm_msgs) if comm_plans is None else comm_plans) * dones_mask

        # --- Coordination ---
        if coord_masks is None:
            # produce mask of coordination from comm_msgs
            coord_masks, glob_rnn_hxs = self.ma_coordinator(current_plans, proc_comm, glob_rnn_hxs)
            if self.training:  # mask here is for `coord_masks` output variable, while for the `modify_qs` step, `proc_comm` is masked
                distance_loss = (-torch.log(torch.clip(torch.abs(coord_masks[..., 0] - coord_masks[..., 1]), 0, 1))).sum() / self.num_agents * 0.1
                coord_masks = F.gumbel_softmax(coord_masks, hard=True, dim=-1) * dones_mask.transpose(1,0).unsqueeze(0)  # add randomness proportional to logits relative value
            blind_coord_masks = torch.argmax(coord_masks, -1, keepdim=True).bool().detach()  # argmax into bool: 0=no coord
        else:
            blind_coord_masks = coord_masks.permute(1,2,0,3)
        # use coordination mask to 'communicate' (selectively allow communication) and improve your choices
        qs = self._modify_qs(solo_qs, rnn_hxs, proc_comm, blind_coord_masks) * dones_mask
        if eval_coord:
            with torch.no_grad():
                if self.eval_coord_mask == "true":
                    inv_qs = self._modify_qs(solo_qs, rnn_hxs, proc_comm,torch.ones_like(blind_coord_masks).bool()) * dones_mask
                elif self.eval_coord_mask == "inverse":
                    inv_qs = self._modify_qs(solo_qs, rnn_hxs, proc_comm, ~blind_coord_masks) * dones_mask
                elif self.eval_coord_mask == "optout":
                    inv_qs = []
                    for i in range(self.num_agents):
                        mask = blind_coord_masks.clone()
                        mask[:, i] = ~mask[:, i]
                        inv_qs.append((self._modify_qs(solo_qs, rnn_hxs, proc_comm, mask) * dones_mask).unsqueeze(2))
                    inv_qs = torch.cat(inv_qs, dim=2)
        else:
            inv_qs = None

        # actions from done agents are not useful in this implementation, so are dropped in the output
        return qs, rnn_hxs, glob_rnn_hxs, inv_qs, coord_masks, (proc_comm, blind_coord_masks.permute(2,0,1,3)), (distance_loss, ae_loss)

    def take_action(self, input, rnn_hxs, glob_rnn_hxs, dones_mask, epsilon):
        """Predict Qs and from those sample an action with a certain temperature"""
        q, rnn_hxs, glob_rnn_hxs, _, _, _, _ = self.forward(input, rnn_hxs, glob_rnn_hxs, dones_mask)
        # sample action to use in the env from q
        actions = self.ma_q.sample_action_from_qs(q, epsilon).squeeze()
        return actions.detach(), rnn_hxs.detach(), glob_rnn_hxs.detach()

    def eval_action(self, input, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans, coord_masks, actions):
        """Off policy call returning Q of given actions"""
        q, rnn_hxs, glob_rnn_hxs, _, _, _, _ = self.forward(input, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans, coord_masks)
        # gather q of action passed
        q_a = q.gather(2, actions.unsqueeze(-1).long()).squeeze(-1)
        return q_a, rnn_hxs, glob_rnn_hxs


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
