from core.ma_gym.comix.comix_modules import *
from utils.utils import TOGETHER, SOLO


class QMixer(nn.Module):
    def __init__(self, observation_space, model_params):
        super(QMixer, self).__init__()
        if model_params is None:
            model_params = {}
        self.num_agents = len(observation_space)
        self.hidden_size = model_params.get("hidden_size", 32)
        self.state_size = self.hidden_size*4

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

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(bs, self.state_size)
        agent_qs = agent_qs.view(bs, 1, self.num_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(bs, self.num_agents, self.hidden_size)
        b1 = b1.view(bs, 1, self.hidden_size)
        hidden = F.relu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.hidden_size, 1)
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(-1, 1)

        return q_tot

    def eval_states(self, states):
        states = states.reshape(-1, self.state_size)
        # Scaling weights used in bmm
        w1 = torch.abs(self.hyper_w_1(states))

        return w1.reshape(-1, self.num_agents, self.hidden_size)


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
        self.action_size = list(self.action_space.values())[0].n  # this breaks action diversity for agents
        self.hidden_size = self.solo_recurrent_size = model_params.get("hidden_size", 64)
        self.coord_recurrent_size = model_params.get("coord_recurrent_size", 256)
        self.shared_comm_ae = model_params.get("ae_comm", False)
        self.cnn_input_proc = model_params.get("cnn_input_proc", False)
        self.eval_coord_mask = model_params.get("eval_coord_mask", False)

        _to_ma = lambda m, args, kwargs: nn.ModuleList([m(*args, **kwargs) for _ in range(self.num_agents)])
        _init_fc_norm = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0))

        # (1) preprocess input to extract relevant features (shared)
        if self.cnn_input_proc:
            self.input_processor = CNNProc(3, self.hidden_size, size=model_params.get("input_proc_size", 0))
        else:
            self.input_processor = FCProc(self.obs_size, self.hidden_size, size=model_params.get("input_proc_size", 0))
        # (2) outgoing communication
        if self.shared_comm_ae:
            self.comm_size = model_params.get("ae_comm_size", 16)
            self.ae = EncoderDecoder(self.hidden_size+self.action_size, self.comm_size)
        else:
            # in theory the msg is formed as concatenation of observation and action
            # i.e. self.comm_size = self.obs_size+self.action_size
            # in practice the message sent is already the final processed_input+actions for sake of computational effort
            self.comm_size = self.hidden_size+self.action_size
        # (3) incoming communication
        self._ids_one_hot = torch.eye(self.num_agents).to(self._device)
        self.plan_size = self.num_agents+self.comm_size  # agent one_hot identifier + hidden + action one_hot

        # Q network to take decisions independently of others
        self.ma_q = QNet(self.agents_ids, self.action_space, self.hidden_size)
        # Produce the coordination mask
        self.ma_coordinator = Coordinator(self.agents_ids, self.action_space, plan_size=self.plan_size, coord_recurrent_size=self.coord_recurrent_size)
        # Coordinated Q network to `slightly` adjust your decisions
        self.co_q_net = _to_ma(nn.GRUCell, (self.plan_size, self.solo_recurrent_size), {})
        self.intent_extractor = nn.Sequential(_init_fc_norm(nn.Linear(self.plan_size, self.solo_recurrent_size)),
                                              nn.ReLU(),
                                              _init_fc_norm(nn.Linear(self.solo_recurrent_size, self.solo_recurrent_size)))
        self.co_q_linear = nn.ModuleList([
            nn.Sequential(_init_fc_norm(nn.Linear(self.solo_recurrent_size*2, self.solo_recurrent_size)),
                          nn.ReLU(),
                          _init_fc_norm(nn.Linear(self.solo_recurrent_size, action_space[id].n))
            ) for id in agents_ids])

        # w init
        for name, param in self.co_q_net.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def get_policy1_parameters(self):
        policy_net_params = []
        policy_net_params += self.input_processor.parameters()
        policy_net_params += self.ma_q.parameters()

        return policy_net_params

    def get_coordinator_parameters(self):
        coordinator_net_params = []
        coordinator_net_params += self.ma_coordinator.parameters()

        return coordinator_net_params

    def get_policy2_parameters(self):
        policy_net_params = []
        policy_net_params += self.intent_extractor.parameters()
        # policy_net_params += self.co_q_net.parameters()
        policy_net_params += self.co_q_linear.parameters()

        return policy_net_params

    def to(self, device):
        self._device = device
        super().to(device)
        # ...the standard recursion applied to submodules set only the modules parameters and not the `_device` variable
        self._ids_one_hot = self._ids_one_hot.to(device)
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
        # Compute q modifications on the base of the coordination induced by the coordination boolean mask
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
            action_logits_coord = action_logits[:, i].clone()
            if not torch.equal(rnn_hxs, hiddens[:, i]):
                action_logits_coord[torch.any((rnn_hxs-hiddens[:, i]).detach(),-1)] += self.co_q_linear[i](rnn_hxs[torch.any((rnn_hxs-hiddens[:, i]).detach(),-1)])
            q_values.append(action_logits_coord.unsqueeze(1))  # NOTE xke questo funzioni il mio hidden deve essere abbastanza informativo da poter inferire gli action_logits precedentemente emessi e di conseguenza con la q una modifica a questi... se non funziona mettere tutto nella q
        q_values = torch.cat(q_values, dim=1)

        return q_values

    def _modify_qs2(self, action_logits, hiddens, comm_plans, masks, temporal_delays=None):
        # Compute q modifications on the base of the coordination induced by the coordination boolean mask
        # `action_logits` (req_grad), `masks` (detached), `comm_msgs` (detached)
        b,n,h = comm_plans.shape
        comm = comm_plans.unsqueeze(1).expand((b, n, n, h))
        intents = torch.zeros(b * n * n, self.solo_recurrent_size, device=self._device)
        masks = masks.permute((2, 0, 1, 3)) * (~torch.eye(n, dtype=torch.bool, device=self._device)).repeat(b, 1, 1).unsqueeze(-1)
        comm = (comm * masks).reshape(b * n * n, -1)
        intents[comm.any(-1)] = self.intent_extractor(comm[comm.any(-1)])
        intents = intents.reshape(b, n, n, -1) / (temporal_delays if temporal_delays is not None else 1)
        intents = intents.sum(-2)

        q_values = []
        for i, id in enumerate(self.agents_ids):
            action_logits_coord = action_logits[:, i].clone()
            if torch.any(intents[:, i]):
                rnn_hxs = hiddens[:, i].clone()
                action_logits_coord += self.co_q_linear[i](torch.cat([rnn_hxs, intents[:, i]], -1))
            q_values.append(action_logits_coord.unsqueeze(1))
        q_values = torch.cat(q_values, dim=1)

        return q_values

    def forward(self, input, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans=None, coord_masks=None, eval_coord=False, tdt=False):
        assert not eval_coord or eval_coord and (comm_plans is None and coord_masks is None), "The arguments combination passed do not match a std wanted behaviour."
        assert not (comm_plans is None and coord_masks is None) or (comm_plans is None and coord_masks is None), "The arguments combination passed is not valid."
        ae_loss = 0
        distance_loss = 0

        # shared input processing
        bs = input.size(0)
        if self.cnn_input_proc:
            input = input.reshape(bs * self.num_agents, *self.obs_shape).transpose(-1, -3)
        else:
            input = input.reshape(bs * self.num_agents, -1)
        x = self.input_processor(input)
        x = x.reshape(bs, self.num_agents, -1)

        # --- (1) Solo action ---
        solo_qs, rnn_hxs = self.ma_q(x, rnn_hxs)
        solo_actions = torch.argmax(solo_qs, dim=-1)
        solo_actions_one_hot = torch.zeros_like(solo_qs).scatter_(-1, solo_actions.type(torch.int64).unsqueeze(-1), 1.)

        # --- (2) Communication ---
        if self.shared_comm_ae:  # TODO should encode both processed obs and id? even if you are done the msg is created
            current_plans = []
            plans = torch.cat([self._ids_one_hot.repeat(bs, 1, 1), x, solo_actions_one_hot], dim=-1)
            for i in range(self.num_agents):
                enc_x, dec_x = self.ae(plans[:, i])
                ae_loss += F.mse_loss(dec_x, plans[:, i])
                current_plans.append((dec_x if self.training else enc_x).unsqueeze(1))  # (*)
            current_plans = torch.cat(current_plans, dim=1).detach()
        else:
            current_plans = torch.cat([self._ids_one_hot.repeat(bs, 1, 1), x, solo_actions_one_hot], dim=-1).detach()  # (**)

        # --- (3) Coordination ---
        # retrieve incoming messages: are the current timestep communicated plans
        comm_msgs = current_plans
        # process incoming messages from others. Also, they need to be masked with dones
        proc_comm = (self._process_msgs(comm_msgs) if comm_plans is None else comm_plans) * dones_mask
        if coord_masks is None:
            # produce mask of coordination from comm_msgs
            coord_masks, glob_rnn_hxs = self.ma_coordinator(current_plans, proc_comm, glob_rnn_hxs)
            done_matrix = dones_mask.permute(1, 2, 0) * dones_mask.permute(2, 1, 0)
            if self.training:  # mask here is for `coord_masks` output variable
                coord_masks = F.gumbel_softmax(coord_masks, hard=True, dim=-1) * done_matrix.unsqueeze(-1)  # add randomness proportional to logits relative value
            blind_coord_masks = (torch.argmax(coord_masks, -1, keepdim=True) * done_matrix.unsqueeze(-1)).bool().detach()  # argmax into bool: 0=no coord
        else:
            blind_coord_masks = coord_masks.permute(1,2,0,3)

        # use coordination mask to 'communicate' (selectively allow communication) and improve your choices
        # temporal_delays = torch.random().to(self._device) if tdt else None
        qs = self._modify_qs2(solo_qs.detach(), rnn_hxs.detach(), proc_comm, blind_coord_masks)
        if eval_coord:
            with torch.no_grad():
                if self.eval_coord_mask == "true":
                    inv_qs = self._modify_qs2(solo_qs.detach(), rnn_hxs.detach(), proc_comm, (torch.ones_like(blind_coord_masks) * done_matrix.unsqueeze(-1)).bool())
                elif self.eval_coord_mask == "inverse":
                    inv_qs = self._modify_qs2(solo_qs.detach(), rnn_hxs.detach(), proc_comm, (~blind_coord_masks * done_matrix.unsqueeze(-1)).bool())
                elif self.eval_coord_mask == "optout":
                    inv_qs = []
                    for i in range(self.num_agents):
                        mask = blind_coord_masks.clone()
                        mask[:, i] = ~mask[:, i]
                        inv_qs.append(self._modify_qs2(solo_qs.detach(), rnn_hxs.detach(), proc_comm, (mask * done_matrix.unsqueeze(-1)).bool()).unsqueeze(2))
                    inv_qs = torch.cat(inv_qs, dim=2)
        else:
            inv_qs = None

        # actions from done agents are not useful in this implementation, so are dropped in the output
        return [solo_qs, qs], rnn_hxs, glob_rnn_hxs, inv_qs, coord_masks, (proc_comm, blind_coord_masks.permute(2,0,1,3)), ae_loss

    def sample_action_from_qs(self, qs):
        """Compute the probability distribution from Q values and sample to obtain the action."""
        # pack batch and agents and sample the distributions
        batch_size, n_agents, _ = qs.shape
        action = torch.multinomial(torch.softmax(qs, dim=-1).view(batch_size*n_agents, -1), 1).view(batch_size, n_agents)
        return action


    def take_action(self, input, rnn_hxs, glob_rnn_hxs, dones_mask):
        """Predict Qs and from those sample an action with a certain temperature."""
        qs, rnn_hxs, glob_rnn_hxs, _, _, _, _ = self.forward(input, rnn_hxs, glob_rnn_hxs, dones_mask)
        # sample action to use in the env from q
        action = self.sample_action_from_qs(qs[TOGETHER]).squeeze()
        return action.detach(), rnn_hxs.detach(), glob_rnn_hxs.detach()

    def eval_action(self, input, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans, coord_masks, actions):
        """Off policy call returning Q of given actions."""
        qs, rnn_hxs, _, _, _, _, _ = self.forward(input, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans, coord_masks)
        # gather q of action passed
        q_a = [qs[SOLO].gather(-1, actions.long()).squeeze(-1), qs[TOGETHER].gather(-1, actions.long()).squeeze(-1)]
        return q_a, rnn_hxs
