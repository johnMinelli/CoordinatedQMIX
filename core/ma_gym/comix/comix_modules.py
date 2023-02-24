import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from core.carla.ppo.model.utils import init, init_normc_

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.init_fc = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        if size == 0:
            self.fc_backbone = nn.Sequential(self.init_fc(nn.Linear(img_num_features, self._hidden_size*4)),
                                    nn.ReLU(),
                                    self.init_fc(nn.Linear(self._hidden_size*4, self._hidden_size*2)),
                                    nn.ReLU(),
                                    self.init_fc(nn.Linear(self._hidden_size*2, self._hidden_size)),
                                    nn.Tanh())
        else:
            self.fc_backbone = nn.Sequential(self.init_fc(nn.Linear(img_num_features, self._hidden_size*2)),
                                    nn.ReLU(),
                                    self.init_fc(nn.Linear(self._hidden_size*2, self._hidden_size)),
                                    nn.Tanh())

    def forward(self, img):
        return self.fc_backbone(img)


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
    def __init__(self, agents_ids, action_space, hidden_size, action_dtype):
        super(QNet, self).__init__()
        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.action_space = action_space
        self.hx_size = hidden_size
        self.action_dtype = action_dtype
        _to_ma = lambda m, args: nn.ModuleList([m(*args) for _ in range(self.num_agents)])

        self.gru = _to_ma(nn.GRUCell, (self.hx_size, self.hx_size))
        self.q = nn.ModuleList([nn.Linear(hidden_size, action_space[id].n) for id in agents_ids])
        for name, param in self.gru.named_parameters():
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
            x = self.gru[i](proc_x[:,i], hidden[:, i])
            next_hidden[i] = x.unsqueeze(1)
            q_values[i] = self.q[i](x).unsqueeze(1)
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action_from_qs(self, out, epsilon):
        mask = torch.rand((out.shape[:2])) <= epsilon
        action = torch.empty((out.shape[0], out.shape[1]), dtype=self.action_dtype).to(self._device)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).type_as(action)
        action[~mask] = out[~mask].argmax(dim=1).type_as(action)

        return action

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = map(lambda o: o.detach().cpu(), self.forward(obs, hidden))
        action = self.sample_action_from_qs(out, epsilon)
        return action, hidden

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.num_agents, self.hx_size)).to(self._device)


class RecurrentHead(nn.Module):
    def __init__(self, hidden_size, recurrent_input_size, bidirectional=False, batch_first=False):
        super(RecurrentHead, self).__init__()

        self.gru = nn.GRU(hidden_size, recurrent_input_size, bidirectional=bidirectional, batch_first=batch_first)  # input features, recurrent steps
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.train()

    def forward(self, x, rnn_hxs, batch_mask):
        # move `padding` (i.e. agents zeroed) at right place then cutted when packing
        compact_seq = torch.zeros_like(x)
        for i, seq_len in enumerate(batch_mask.sum(0)):  # cycle over the batch and sum over agents
            compact_seq[:seq_len, i] = x[batch_mask[:,i],i]
        # pack in sequence dimension (the number of agents)
        packed_x = pack_padded_sequence(compact_seq, batch_mask.sum(0).cpu().numpy(), enforce_sorted=False)
        packed_scores, rnn_hxs = self.gru(packed_x, rnn_hxs)
        scores, _ = pad_packed_sequence(packed_scores)  # restore sequence dimension (the number of agents)
        # restore order moving padding in its place
        scores = torch.zeros((*batch_mask.shape,scores.size(-1))).to(scores.device).masked_scatter(batch_mask.unsqueeze(-1), scores)
        return scores, rnn_hxs


class Coordinator(nn.Module):
    """Depending on the result of the BiGru module the action returned by the policy
     will be used or resampled with aid of communication messages"""
    def __init__(self, agents_ids, action_space, plan_size, solo_recurrent_size, coord_recurrent_size):
        super(Coordinator, self).__init__()
        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.agents_ids = agents_ids
        self.action_space = action_space
        self.plan_size = plan_size
        self.recurrent_size = coord_recurrent_size
        _to_ma = lambda m, args, kwargs: nn.ModuleList([m(*args, **kwargs) for _ in range(self.num_agents)])
        _init_fc_ortho = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.global_coord_net = _to_ma(RecurrentHead, (self.plan_size*2, coord_recurrent_size), {"bidirectional": True, "batch_first": False})  # [self, others]
        self.boolean_coordinator = nn.ModuleList([nn.Sequential(
            _init_fc_ortho(nn.Linear(coord_recurrent_size*2, coord_recurrent_size)),  # *2 because takes the bidirectional hxs of global_coord_net
            nn.ReLU(),
            _init_fc_ortho(nn.Linear(coord_recurrent_size, 2))) for id in agents_ids])

    def to(self, device):
        self._device = device
        super().to(device)
        return self

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_agents, 2, batch_size, self.recurrent_size)).to(self._device)

    def forward(self, plans, comm_plans, glob_hiddens):
        """
        :param action_logits: (req_grad) selfish action logits to modify
        :param plans: (detached) current selfish_plans as starting points
        :param comm_plans: (detached for ae / req_grad for fc) other's agents plans to mix with current selfish plans (except self comm_plan)
        :param hiddens: (req_grad) selfish hiddens
        :param glob_hiddens: (req_grad) global hiddens (2n,n,b,h)
        :param eval_coord: compute policy value also for the opposite mask
        """
        # The coordination part is detached from the rest: it produces only the boolean coord_masks
        glob_rnn_hxs = []
        coord_masks = []
        comm_plans[~torch.any(torch.any(comm_plans,-1),-1)] = plans[~torch.any(torch.any(comm_plans,-1),-1)]  # patch for t=0 case
        glob_batch_mask = torch.any(comm_plans,-1).transpose(0,1)  # (n,b)
        for i in range(self.num_agents):
            # if not torch.any(plans[:,i]): continue  # agent done
            # others_plans = torch.cat((comm_plans[:, :i], plans[:,i].unsqueeze(1), comm_plans[:, i+1:]), dim=1).detach()  # TODO should I inject the prev decision of communicate? should I make the coordinator conscious of who is coordinating for?
            others_plans = torch.cat([plans[:,i].unsqueeze(1).expand(comm_plans.shape), comm_plans],-1)
            x = others_plans.transpose(0, 1)  # (n,b,h)
            scores, rnn_hxs = self.global_coord_net[i](x, glob_hiddens[i], glob_batch_mask)
            scores = scores.transpose(0, 1)
            glob_rnn_hxs.append(rnn_hxs.unsqueeze(0))  # TODO io uso la GRUCell con sequenze di n_agents, ma l'hidden riesce a capire la ricorrenza delle decisioni a gruppi di n_agents? intrarelazioni in sequenza ed interrelazioni con la decisione finale (se la decisione finale parla dell'agente alla posizione x della sequenza che alla volta dopo sarà sempre alla posizione x)?

            coord_mask = []
            for j in range(scores.size(1)):  # to optimize by reshape
                coord_mask.append(self.boolean_coordinator[i](scores[:,j]).unsqueeze(0))
            coord_masks.append(torch.cat(coord_mask, dim=0).unsqueeze(0))
        glob_rnn_hxs = torch.cat(glob_rnn_hxs, dim=0)
        coord_masks = torch.cat(coord_masks, dim=0)

        return coord_masks, glob_rnn_hxs
