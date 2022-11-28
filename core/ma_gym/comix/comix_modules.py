from core.carla.ppo.model.utils import init, init_normc_

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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
        _to_ma = lambda m: nn.ModuleList([m for _ in range(self.num_agents)])

        self.gru = _to_ma(nn.GRUCell(self.hx_size, self.hx_size))
        self.q = nn.ModuleList([nn.Linear(hidden_size, action_space[id].n) for id in agents_ids])

    def to(self, device):
        self._device = device
        super().to(device)
        return self

    def forward(self, proc_x, hidden, dones):
        batch_s = proc_x.shape[0]
        q_values = [torch.empty(batch_s, )] * self.num_agents
        next_hidden = [torch.empty(batch_s, 1, self.hx_size)] * self.num_agents
        for i in range(self.num_agents):
            x = self.gru[i](proc_x[:,i], hidden[:, i])
            next_hidden[i] = x.unsqueeze(1)
            q_values[i] = self.q[i](x).unsqueeze(1)
        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action_from_qs(self, out, epsilon):
        mask = torch.rand((out.shape[0])) <= epsilon
        action = torch.empty((out.shape[0], out.shape[1]), dtype=self.action_dtype).to(self._device)
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).type_as(action)
        action[~mask] = out[~mask].argmax(dim=2).type_as(action)

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

    def forward(self, x, rnn_hxs):
        x, rnn_hxs = self.gru(x, rnn_hxs)
        return x, rnn_hxs


class Coordinator(nn.Module):
    """Depending on the result of the BiGru module the action returned by the policy
     will be used or resampled with aid of communication messages"""
    def __init__(self, agents_ids, action_space, plan_size, recurrent_size):
        super(Coordinator, self).__init__()
        self._device = torch.device("cpu")
        self.num_agents = len(agents_ids)
        self.agents_ids = agents_ids
        self.action_space = action_space
        self.plan_size = plan_size
        self.recurrent_size = recurrent_size
        _to_ma = lambda m: nn.ModuleList([m for _ in range(self.num_agents)])

        self.global_coord_net = _to_ma(RecurrentHead(plan_size, recurrent_size, bidirectional=True, batch_first=False))
        self.boolean_coordinator = _to_ma(nn.Sequential(
            nn.Linear(recurrent_size*2, recurrent_size),  # *2 because takes the bidirectional hxs of global_coord_net
            nn.ReLU(),
            nn.Linear(recurrent_size, 2),
            nn.Softmax()))

        # The role of these modules is equivalent to a QNet but are created as it is for granularity of usage
        self.coord_net = nn.ModuleList([nn.GRUCell(plan_size-action_space[id].n, plan_size-action_space[id].n) for id in agents_ids])
        self.q = nn.ModuleList([nn.Linear(plan_size-action_space[id].n, action_space[id].n) for id in agents_ids])

    def to(self, device):
        self._device = device
        super().to(device)
        return self

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_agents, 2, batch_size, self.recurrent_size)).to(self._device)

    def forward(self, action_logits, plans, comm_plans, hiddens, glob_hiddens, dones, eval_coord):
        """
        :param action_logits: (req_grad) selfish action logits to modify
        :param plans: (detached) current selfish_plans as starting points
        :param comm_plans: (detached for ae / req_grad for fc) other's agents plans to mix with current selfish plans (except self comm_plan)
        :param hiddens: (req_grad) selfish hiddens
        :param glob_hiddens: (req_grad) global hiddens
        :param dones: (detached) terminated agents masks
        :param eval_coord: compute policy value also for the opposite mask
        :return: coordianted actions between all agents, masks of coordination
        """
        # Coordination part detached from the rest: it produces only the boolean coord_masks
        glob_rnn_hxs = []
        coord_masks = []
        for i in range(self.num_agents):
            # if not torch.any(plans[:,i]): continue  # agent done
            others_plans = torch.cat((comm_plans[:, :i], plans[:,i].unsqueeze(1), comm_plans[:, i+1:]), dim=1).detach()  # should i inject the prev decision of communicate? 
            x = others_plans.transpose(0, 1)
            scores, rnn_hxs = self.global_coord_net[i](x, glob_hiddens[i])
            scores = scores.transpose(0,1)
            glob_rnn_hxs.append(rnn_hxs.unsqueeze(0))

            coord_mask = []
            for j in range(scores.size(1)):
                coord_mask.append(self.boolean_coordinator[i](scores[:,j]).unsqueeze(0))
            coord_masks.append(torch.cat(coord_mask, dim=0).unsqueeze(0))
        glob_rnn_hxs = torch.cat(glob_rnn_hxs, dim=0)
        coord_masks = torch.cat(coord_masks, dim=0)

        # Compute q modification on the base of the forced coordination induced by the coord_mask
        q_values = None  # with the selfish `action_logits` it will allow the propagation of gradients for the policy
        inv_q_values = None  # this will criticize the mask creation
        # `action_logits` (req_grad), `masks` (detached), `comm_msgs` (detached)
        # TODO come nuovo hidden mantengo quello vecchio in solo, oppure lo aggiorno con la coordinazione?
        blind_coord_masks = torch.argmax(coord_masks, -1, keepdim=True).detach()
        q_values = []
        for i, id in enumerate(self.agents_ids):
            rnn_hxs = hiddens[:,i].clone()
            for j in range(self.num_agents):
                if i == j: continue
                comm_plans_masked = comm_plans[:,j,:-self.action_space[id].n] * blind_coord_masks[i, j]
                rnn_hxs = self.coord_net[i](comm_plans_masked, rnn_hxs)
            q_values.append((action_logits[:,i]+self.q[i](rnn_hxs)).unsqueeze(1))  # NOTE xke questo funzioni il mio hidden deve essere abbastanza informativo da poter inferire gli action_logits precedentemente emessi e di conseguenza con la q una modifica a questi... se non funziona mettere tutto nella q
        q_values = torch.cat(q_values, dim=1)

        # Compute q with opposite mask predictions for coordinator loss
        if eval_coord:
            with torch.no_grad():
                inv_q_values = []
                for i, id in enumerate(self.agents_ids):
                    rnn_hxs = hiddens[:, i]
                    for j in range(self.num_agents):
                        if i == j: continue
                        comm_plans_masked = comm_plans[:, j, :-self.action_space[id].n] * ~blind_coord_masks[i, j]
                        rnn_hxs = self.coord_net[i](comm_plans_masked, rnn_hxs)
                    inv_q_values.append((action_logits[:, i] + self.q[i](rnn_hxs)).unsqueeze(1))
                inv_q_values = torch.cat(inv_q_values, dim=1)
        else: inv_q_values = None

        return q_values, inv_q_values, glob_rnn_hxs, coord_masks
