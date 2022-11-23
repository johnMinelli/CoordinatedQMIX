from tensorflow.python.distribute.device_util import current

from core.carla.ppo.model.distributions import Categorical, DiagGaussian
from core.carla.ppo.model.utils import init, init_normc_
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


class NonePlaceholder(nn.Module):
    def forward(self, x):
        return x, None


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class MixNet(nn.Module):
    def __init__(self, observation_space, hidden_dim=32, hx_size=64, recurrent=False):
        super(MixNet, self).__init__()

        self._device = torch.device("cpu")
        state_size = sum([_.shape[0] for _ in observation_space])
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


class Coordinator(nn.Module):
    """Depending on the result of the BiGru module the action returned by the policy
     will be used or resampled with aid of communication messages"""
    def __init__(self, agents_ids, hidden_size, recurrent_input_size):

        self.num_agents = len(agents_ids)
        _to_ma = lambda m: nn.ModuleList([m for _ in range(self.num_agents)])

        self.global_coord_net = _to_ma(nn.GRU(hidden_size, recurrent_input_size, bidirectional=True))
        self.boolean_coordination = _to_ma(nn.Sequential(
            nn.Linear(recurrent_input_size, recurrent_input_size//2),
            nn.ReLU(),
            nn.Linear(recurrent_input_size//2, 2),
            nn.Sigmoid()))

    def forward(self, action_logits, plans, comm_plans, hiddens):
        """
        :param action_logits: selfish action logits to modify
        :param plans: current selfish_plans as starting points
        :param comm_plans: other's agents plans to mix with current selfish plans (except self comm_plan)
        :param hiddens: selfish hiddens  # TODO unsure about it
        :return: coordianted actions between all agents, masks of coordination
        """
        coord_masks = []
        for i in range(self.agents):
            others_plans = comm_plans - i  # TODO
            x = torch.cat([plans[i], others_plans], dim=1)
            scores, hidden = self.global_coord_net[i](x, hiddens[i])
            coord_mask = []
            for score in scores:
                coord_mask.append(self.boolean_coordination[i](score))
            coord_masks.append(torch.cat(coord_mask, dim=0))

        # action logits (active), mask (detached), comm_msgs (detached)
        blind_mask = [m.detach() for m in coord_mask]
        # TODO GruCell to mix (make my policy be influenced by others) with unmasked actors? use `x` and `blind_mask`
        masked_x = x
        x = action_logits+self.policyadded(masked_x)
        # TODO actions selection
        actions = None

        return actions, coord_mask


class Policy(nn.Module):
    """
    Directly predict n values to sample from a learnt distribution: Categorical if Discrete action space else Gaussian
    The output size is dependent to the action size of the environment
    """
    def __init__(self, agents_ids, observation_space, action_space, ma_coordinate, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.num_agents = len(agents_ids)
        self.obs_size = np.prod(observation_space[id].shape)
        self.action_space_size = list(action_space.values())[0].n
        self.action_dtype = torch.int if list(action_space.values())[0].__class__.__name__ == 'Discrete' else torch.float32
        self.ma_coordinate = ma_coordinate
        self.hidden_size = base_kwargs["hidden_size"]  # 64
        self.recurrent_input_size = base_kwargs["recurrent_input_size"]
        self.shared_encoder = True
        self.cnn_proc = True
        _to_ma = lambda m: nn.ModuleList([m for _ in range(self.num_agents)])
        # handle implementation with or without vocabulary of measurements

        # preprocess input
        if self.cnn_proc:
            self.input_processor = CNNProc(self.obs_size, self.hidden_size)
        else:
            self.input_processor = FCProc(self.obs_size, self.hidden_size)
        # to handle the communication messages
        if self.shared_encoder:
            self.comm_size = base_kwargs["comm_size"]  # 10
            self.ae = EncoderDecoder(self.hidden_size, self.comm_size)
        else:
            self.comm_size = self.obs_size
            self.fc_processor = _to_ma(nn.Sequential(self.init_fc(nn.Linear(self.comm_size, self.hidden_size * 4)), nn.ReLU(),
                self.init_fc(nn.Linear(self.hidden_size * 4, self.hidden_size * 2)), nn.ReLU(),
                self.init_fc(nn.Linear(self.hidden_size * 2, self.hidden_size)), nn.ReLU()))
        # recurrent policy to get the selfish action
        self.ma_policy = QNet(self.hidden_size, self.action_space_size, self.action_dtype)  # 64 + maybe also my encoding? in that case add encoding

        if self.ma_coordinate:
            self.ma_coordinator = Coordinator(self.hidden_size, self.recurrent_input_size)

    @property
    def is_recurrent(self):
        return True

    def forward(self, input, rnn_hxs, comm, masks):
        ae_loss = 0
        # shared
        x = torch.cat(input, dim=0)
        x = self.input_processor(x)
        x = torch.chunk(x, self.num_agents)

        logits, rnn_hxs = self.ma_policy(x, rnn_hxs)
        actions = self.ma_policy.sample_action_from_logits(logits)

        # create self comm_msg: should encode both processed obs and actions?
        if self.shared_encoder:
            current_plans = torch.cat([x, actions], dim=1)
            for i in range(self.num_agents):
                enc_x = self.ae(current_plans[i])
                dec_x = self.decoder(enc_x)
                ae_loss += F.mse_loss(dec_x, current_plans[i])
                current_plans[i] = (dec_x if not self.eval() else enc_x).detach()  # (*)
        else:  # alternatively pass the entire observation
            current_plans = torch.cat([input, actions], dim=1)

        # process other's comm_msg
        proc_comm = []
        if self.shared_encoder:
            if self.eval():
                # (*) being off-policy we cannot give a loss on the base of the decoding
                # of past encoded messages so while training the messages will be sent already decoded
                for i in range(self.num_agents):
                    cf = self.ae.decode(comm[i])
                    # if not self.ae_pg:
                    #     cf = cf.detach()
                    proc_comm.append(cf)
            else: proc_comm = comm
        else:
            for i in range(self.num_agents):
                proc_comm = self.fc_processor[i](comm[i])
            proc_comm = torch.cat(proc_comm, dim=0)

        actions, coordinators_mask = self.ma_coordinator(logits, current_plans, proc_comm, rnn_hxs.detach())

        return actions, rnn_hxs, current_plans, coordinators_mask, ae_loss


class EncoderDecoder(nn.Module):
    def __init__(self, comm_len, processed_size=64):
        super(EncoderDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(processed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, comm_len),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(comm_len, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, processed_size),
        )

    def decode(self, x):
        return self.decoder(x)  # (num_agents, processed_size)

    def forward(self, feat):
        return self.encoder(feat)


class CNNProc(nn.Module):
    def __init__(self, img_num_features, hidden_size=512):
        super(CNNProc, self).__init__()

        self._hidden_size = hidden_size
        self.init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        self.cnn_backbone = nn.Sequential(
            self.init_cnn(nn.Conv2d(img_num_features, 32, 8, stride=4)),
            nn.ReLU(),
            self.init_cnn(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.init_cnn(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            self.init_cnn(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

    def forward(self, img):
        return self.cnn_backbone(img)


class FCProc(nn.Module):
    def __init__(self, img_num_features, hidden_size=32):
        super(FCProc, self).__init__()

        self._hidden_size = hidden_size

        self.fc_backbone = nn.Sequential(nn.Linear(img_num_features, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, self._hidden_size),
                                nn.ReLU())

    def forward(self, img):
        return self.fc_backbone(img)


class QNet(nn.Module):
    def __init__(self, agents_ids, hidden_size, action_size, action_dtype, recurrent=True):
        super(QNet, self).__init__()

        self.num_agents = len(agents_ids)
        self.hx_size = hidden_size
        self.out_size = action_size
        self.recurrent = recurrent
        self.action_dtype = action_dtype
        _to_ma = lambda m: nn.ModuleList([m for _ in range(self.num_agents)])

        for i, id in enumerate(agents_ids):
            if recurrent:
                self.gru = self._to_ma(nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(i), nn.Linear(self.hx_size, self.out_size))

    def forward(self, proc_x, hidden):
        batch_s = proc_x.shape[0]
        q_values = [torch.empty(batch_s, )] * self.num_agents
        next_hidden = [torch.empty(batch_s, 1, self.hx_size)] * self.num_agents
        for i in range(self.num_agents):
            if self.recurrent:
                x = self.gru[i](proc_x, hidden[:, i, :])
                next_hidden[i] = x.unsqueeze(1)
            else: x = proc_x
            q_values[i] = self.q[i](x).unsqueeze(1)
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = map(lambda o: o.detach().cpu(), self.forward(obs, hidden))
        mask = (torch.rand((out.shape[0])) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1]), dtype=self.action_dtype)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).type_as(action)
        action[~mask] = out[~mask].argmax(dim=2).type_as(action)
        return action, hidden

    def sample_action_from_logits(self, obs, hidden, epsilon):
        out, hidden = map(lambda o: o.detach().cpu(), self.forward(obs, hidden))
        mask = (torch.rand((out.shape[0])) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1]), dtype=self.action_dtype)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).type_as(action)
        action[~mask] = out[~mask].argmax(dim=2).type_as(action)
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))


class RecurrentHead(nn.Module):
    def __init__(self, hidden_size, recurrent_input_size):
        self._recurrent_input_size = recurrent_input_size

        self.gru = nn.GRU(hidden_size, recurrent_input_size)  # input features, recurrent steps
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.critic_linear = self.init_fc(nn.Linear(recurrent_input_size, 1))
        self.train()

# NOTE:
#  - dubbio sugli encoder di `measures` e `communication`: così tanti Linear?
#  - dubbio su come il comm venga sommato all' x in input al GRU: magari in input all' fc_joint?

    @property
    def output_size(self):
        return self._recurrent_input_size

    def to(self, device):
        self.gru.to(device)
        super().to(device)

    def forward(self, x, rnn_hxs, done_masks):

        if x.size(0) == rnn_hxs.size(0):  # batch size contains whole sequences?
            x, rnn_hxs = self.gru(x.unsqueeze(0), (rnn_hxs * done_masks).unsqueeze(0))  # unsqueeze to set num_layers dim=1
            x = x.squeeze(0)
            rnn_hxs = rnn_hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = rnn_hxs.size(0)  # batch
            T = int(x.size(0) / N)  # seq length

            # unflatten
            x = x.view(T, N, x.size(1))
            done_masks = done_masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (done_masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the done_masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            rnn_hxs = rnn_hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in done_masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, rnn_hxs = self.gru(
                    x[start_idx:end_idx],
                    rnn_hxs * done_masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            rnn_hxs = rnn_hxs.squeeze(0)

        logits = x
        return self.critic_linear(logits), logits, rnn_hxs

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
        self.q_policy = Policy(env.observation_space, env.action_space, self.opt.recurrent, ma_coordinate=True)
        self.q_policy_target = Policy(env.observation_space, env.action_space, self.opt.recurrent, ma_coordinate=False)
        self.q_policy_target.load_state_dict(self.q_policy.state_dict())
        self.mix_net = MixNet(env.observation_space, recurrent=self.opt.recurrent)
        self.mix_net_target = MixNet(env.observation_space, recurrent=self.opt.recurrent)
        self.mix_net_target.load_state_dict(self.mix_net.state_dict())
        self.memory = ReplayBuffer(opt.max_buffer_len)

        # Load or init
        if not opt.isTrain or opt.continue_train is not None:
            if opt.isTrain:
                self.start_epoch = self.load_model(self.backup_dir, "policy", opt.continue_train)
            else:
                self.load_model(opt.models_path, "policy", opt.model_epoch)
        # else:
        #     init_weights(self.model, init_type="normal")

        if opt.isTrain:  # TODO who trains what?
            self.q_policy.train()
            self.mix_net.train()
            # initialize optimizers
            self.schedulers, self.optimizers = [], []
            self.optimizer_ae = optim.Adam(self.q_policy.ae.parameters(), lr=opt.lr)
            self.optimizer_G = optim.Adam(self.q_policy.parameters() + self.mix_net.parameters(), lr=opt.lr)

            self.optimizers.append(self.optimizer_G)

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

    def take_action(self, observation, hidden=None, comm=None, explore=True):
        actions, hidden, comm, _, ae_loss = self.q_policy.take_action(observation, hidden, comm, self.epsilon if explore else 0)
        # optimize the ae while acting
        self.optimizer_ae.zero_grad()
        ae_loss.backward()
        self.optimizer_ae.step()

        return actions, hidden, comm

    def step(self, state, comm, action, reward, next_state, next_comm, done):

        self.memory.put((state, comm, action, (np.array(reward)).tolist(), next_state, next_comm, np.array(done, dtype=int).tolist()))
        losses = {}

        if self.memory.size() > self.warm_up_steps:
            q_loss_epoch = 0

            _chunk_size = self.chunk_size
            for _ in range(self.K_epochs):
                states, comm, actions, rewards, next_states, next_comm, dones = self.memory.sample_chunk(self.batch_size, _chunk_size)
                hidden = self.q_policy.init_hidden(self.batch_size).to(self.device)
                target_hidden = self.q_policy_target.init_hidden(self.batch_size).to(self.device)
                mix_net_target_hidden = self.mix_net_target.init_hidden(self.batch_size)
                mix_net_hidden = [torch.empty_like(mix_net_target_hidden) for _ in range(_chunk_size + 1)]
                mix_net_hidden[0] = self.mix_net_target.init_hidden(self.batch_size).to(self.device)

                loss = 0
                for step_i in range(_chunk_size):
                    q_out, hidden, comm, coord_masks, _ = self.q_policy(states[:, step_i], hidden, comm[:, step_i])
                    q_a = q_out.gather(2, actions[:, step_i].unsqueeze(-1).long()).squeeze(-1)
                    pred_q, next_mix_net_hidden = self.mix_net(q_a, states[:, step_i], mix_net_hidden[step_i])

                    max_q_target, target_hidden = self.q_policy_target(next_states[:, step_i], target_hidden.detach(), next_comm[:, step_i])
                    max_q_target = max_q_target.max(dim=2)[0].squeeze(-1)
                    next_q_total, mix_net_target_hidden = self.mix_net_target(max_q_target, next_states[:, step_i, :, :],
                                                                               mix_net_target_hidden.detach())
                    target = rewards[:, step_i, :].sum(dim=1, keepdims=True) + (self.gamma * next_q_total * (1 - done[:, step_i]))
                    loss += F.smooth_l1_loss(pred_q, target.detach())

                    done_mask = done[:, step_i].squeeze(-1).bool()
                    hidden[done_mask] = self.q_policy.init_hidden(len(hidden[done_mask]))
                    target_hidden[done_mask] = self.q_policy_target.init_hidden(len(target_hidden[done_mask])).to(self.device)
                    mix_net_hidden[step_i + 1][~done_mask] = next_mix_net_hidden[~done_mask]
                    # (len(hidden))[done_mask].view(*hidden[done_mask].shape)
                    mix_net_hidden[step_i + 1][done_mask] = self.mix_net.init_hidden(len(mix_net_hidden[step_i][done_mask])).to(self.device)
                    mix_net_target_hidden[done_mask] = self.mix_net_target.init_hidden(len(mix_net_target_hidden[done_mask])).to(self.device)

                self.optimizer_G.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_policy.parameters(), self.grad_clip_norm, norm_type=2)
                torch.nn.utils.clip_grad_norm_(self.mix_net.parameters(), self.grad_clip_norm, norm_type=2)
                self.optimizer_G.step()

                q_loss_epoch += loss.item()

            num_updates = self.K_epochs * _chunk_size
            q_loss_epoch /= num_updates
            losses = {"value_loss": q_loss_epoch}

        return losses

    def update_target(self):
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




class Others():
    def __init__(self):
## mixer part 
        self.init_fc = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        if msr_num_features != 0:


            self.fc_joint = nn.Sequential(
                self.init_fc(nn.Linear(hidden_size*3, hidden_size*2)),
                nn.ReLU(),
                self.init_fc(nn.Linear(hidden_size*2, hidden_size)),
                nn.ReLU(),
                self.init_fc(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU()
            )

        self.critic_linear = self.init_fc(nn.Linear(hidden_size, 1))
        self.train()
## mixer part


    def _backbone_proc(self, img, msr, comm):
        x_img = self.cnn_backbone(img)

        # if msr is not None:  # TODO
        x_msr = self.fc_enc(msr)
        # x_comm = self.comm_enc(comm)
        # x = torch.cat([x_img, x_msr, comm], -1)
        # x = self.fc_joint(x)
        x = torch.cat([x_img, x_msr, comm], -1)
        x = self.fc_joint(x)
        # else:
        #     x = x_img
        return x