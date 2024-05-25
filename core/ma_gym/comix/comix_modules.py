from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from core.ma_gym.utils.utils import init

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNProc(nn.Module):
    def __init__(self, img_num_features, hidden_size=512, size=0):
        super(CNNProc, self).__init__()

        self._hidden_size = hidden_size
        self.init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        if size == 0:
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
        else:
            self.cnn_backbone = nn.Sequential(
                self.init_cnn(nn.Conv2d(img_num_features, 32, 3, stride=1)),
                nn.ReLU(),
                self.init_cnn(nn.Conv2d(32, 64, 3, stride=1)),
                nn.MaxPool2d(3),
                nn.ReLU(),
                Flatten(),
                self.init_cnn(nn.Linear(64, hidden_size)),
                nn.Tanh()
            )

    def forward(self, img):
        return self.cnn_backbone(img)


class FCProc(nn.Module):
    def __init__(self, img_num_features, hidden_size=32, size=0):
        super(FCProc, self).__init__()

        self._hidden_size = hidden_size
        self.init_fc = lambda m: init(m, lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'), lambda x: nn.init.constant_(x, 0))

        if size == 0:
            self.fc_backbone = nn.Sequential(self.init_fc(nn.Linear(img_num_features, self._hidden_size*4)),
                                    nn.ReLU(),
                                    self.init_fc(nn.Linear(self._hidden_size*4, self._hidden_size*2)),
                                    nn.ReLU(),
                                    self.init_fc(nn.Linear(self._hidden_size*2, self._hidden_size)),
                                    nn.ReLU())
        else:
            self.fc_backbone = nn.Sequential(self.init_fc(nn.Linear(img_num_features, self._hidden_size*2)),
                                    nn.ReLU(),
                                    self.init_fc(nn.Linear(self._hidden_size*2, self._hidden_size)),
                                    nn.ReLU())

    def forward(self, state):
        return self.fc_backbone(state)


class EncoderDecoder(nn.Module):
    def __init__(self, processed_size=64, comm_len=10):
        super(EncoderDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(processed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, comm_len),
            nn.Tanh()
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

    def forward(self, feat):
        enc_x = self.encoder(feat)
        dec_x = self.decoder(enc_x)  # (num_agents, processed_size)
        return enc_x, dec_x

    def decode(self, x):
        return self.decoder(x)


class QNet(nn.Module):
    """Network implementation for multi-agent"""
    def __init__(self, agents_ids, action_space, hidden_size, ln):
        super(QNet, self).__init__()
        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.action_space = action_space
        self.hx_size = hidden_size
        self.use_ln = ln
        _to_ma = lambda m, args: nn.ModuleList([m(*args) for _ in range(self.num_agents)])

        self.gru_modules = _to_ma(nn.GRUCell, (self.hx_size, self.hx_size))
        self.ln_modules = _to_ma(nn.LayerNorm, (self.hx_size,))
        self.q_modules = nn.ModuleList([nn.Linear(hidden_size, action_space[id].n) for id in agents_ids])
        for name, param in self.gru_modules.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def to(self, device):
        self._device = device
        super().to(device)
        return self

    def forward(self, proc_x, hidden):
        batch_s = proc_x.shape[0]
        q_values = [torch.empty(batch_s, )] * self.num_agents
        next_hidden = [torch.empty(batch_s, 1, self.hx_size)] * self.num_agents
        for i in range(self.num_agents):
            x = self.gru_modules[i](proc_x[:, i], hidden[:, i])
            next_hidden[i] = x.unsqueeze(1)
            # x = x+proc_x[:, i]  # skip
            if self.use_ln:
                x = self.ln_modules[i](x)
            q_values[i] = self.q_modules[i](x).unsqueeze(1)
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.num_agents, self.hx_size)).to(self._device)


class RecurrentHead(nn.Module):
    def __init__(self, hidden_size, recurrent_input_size, bidirectional=False, batch_first=False):
        super(RecurrentHead, self).__init__()

        self.gru = nn.GRU(hidden_size, recurrent_input_size, bidirectional=bidirectional, batch_first=batch_first)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.train()

    def forward(self, x, rnn_hxs, batch_mask):
        """Note: we don't want to the GRU to handle empty messages otw the seq in large envs would be mostly filled with 0s.
            The packing procedure solve this problem by using batch_mask to align all sent messages to left side ignoring empty messages.
            After the computation through the recurrent unit the scores relative a message are scattered back in the original
            position relative to agent.
            e.g. mask [T,F,F,T] and msgs [m0, m1, m2, m3] --> packed msgs [m0, m3, -, -] = [s0, s1, , -] --> scattered scores [s0, 0, 0, s1]
        """
        # move `padding` (i.e. agents zeroed) at right place then cutted when packing
        compact_seq = torch.zeros_like(x)
        seq_lengths = batch_mask.sum(0)
        packed_mask = torch.arange(x.size(0)).reshape(-1, 1).to(x.device) < seq_lengths.reshape(1, -1)
        compact_seq[packed_mask, :] = x[batch_mask, :]
        # pack in sequence dimension (the number of agents)
        packed_x = pack_padded_sequence(compact_seq, seq_lengths.cpu().numpy(), enforce_sorted=False)
        packed_scores, rnn_hxs = self.gru(packed_x, rnn_hxs)
        scores, _ = pad_packed_sequence(packed_scores)  # restore sequence dimension (the number of agents)
        # restore order moving padding in its place
        scores = torch.zeros((*batch_mask.shape,scores.size(-1))).to(scores.device).masked_scatter(batch_mask.unsqueeze(-1), scores)
        return scores, rnn_hxs


class Coordinator(nn.Module):
    """Depending on the result of the BiGru module the action returned by the policy
     will be used or resampled with aid of communication messages"""
    def __init__(self, agents_ids, num_dummy_agents, action_space, plan_size, coord_recurrent_size, ln):
        super(Coordinator, self).__init__()
        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.num_agents_dummy = num_dummy_agents
        self.agents_ids = agents_ids
        self.action_space = action_space
        self.plan_size = plan_size
        self.recurrent_size = coord_recurrent_size
        self.use_ln = ln
        _to_ma = lambda m, args, kwargs: nn.ModuleList([m(*args, **kwargs) for _ in range(self.num_agents)])
        _init_fc = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0))

        self.coord_net_modules = _to_ma(RecurrentHead, (self.plan_size * 2, coord_recurrent_size), {"bidirectional": True, "batch_first": False})  # it takes as input seq [[self, other],...]
        self.ln_modules = _to_ma(nn.LayerNorm, (coord_recurrent_size*2,), {})
        self.bool_coord_modules = nn.ModuleList([
            nn.Sequential(
                _init_fc(nn.Linear(coord_recurrent_size*2, coord_recurrent_size)),  # *2 since it takes the bidirectional hxs of coord_net_modules[i]
                nn.ReLU(),
                _init_fc(nn.Linear(coord_recurrent_size, 2))
            ) for id in agents_ids])

    def to(self, device):
        self._device = device
        super().to(device)
        return self

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_agents, 2, batch_size, self.recurrent_size)).to(self._device)

    def forward(self, plans, comm_plans, coord_hiddens):
        """
        :param plans: (detached) current selfish_plans as starting points
        :param comm_plans: (detached for ae / req_grad for fc) other's agents plans to mix with current selfish plans (except self comm_plan)
        :param hiddens: (req_grad) selfish hiddens
        :param coord_hiddens: (req_grad) global coordinator hiddens (2n,n,b,h)
        """
        # The coordination part is detached from the rest: it produces only the boolean coord_masks
        coord_rnn_hxs = []
        coord_masks = []
        comm_plans[~torch.any(torch.any(comm_plans,-1),-1)] = plans[~torch.any(torch.any(comm_plans,-1),-1)]  # patch: filling comm_channel with in domain data in case of a batch sample with no comm (e.g. for t=0 case when there is a delay between msg send and reception)
        agents_communicating_batch_mask = torch.any(comm_plans,-1).transpose(0,1)  # (n,b)
        agents_communicating_batch_mask = torch.cat([agents_communicating_batch_mask, torch.ones((self.num_agents_dummy-agents_communicating_batch_mask.size(0), *agents_communicating_batch_mask.shape[1:]), device=agents_communicating_batch_mask.device)]).bool()
        for i in range(self.num_agents):
            # if not torch.any(plans[:,i]): continue  # agent done
            others_plans = torch.cat([plans[:,i].unsqueeze(1).expand(comm_plans.shape), comm_plans],-1)
            x = others_plans.transpose(0, 1)  # (n,b,h)
            x = torch.cat([x, torch.randn((self.num_agents_dummy-x.size(0), *x.shape[1:]), device=agents_communicating_batch_mask.device)])
            scores, rnn_hxs = self.coord_net_modules[i](x, coord_hiddens[i], agents_communicating_batch_mask)
            if self.use_ln:
                scores = self.ln_modules[i](scores)
            agent_coord_masks = self.bool_coord_modules[i](scores)

            coord_rnn_hxs.append(rnn_hxs.unsqueeze(0))
            coord_masks.append(agent_coord_masks.unsqueeze(0))
        coord_rnn_hxs = torch.cat(coord_rnn_hxs, dim=0)
        coord_masks = torch.cat(coord_masks, dim=0)

        return coord_masks, coord_rnn_hxs
