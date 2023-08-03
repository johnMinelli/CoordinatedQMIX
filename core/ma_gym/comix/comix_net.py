import torch.nn.functional as F

from core.ma_gym.comix.comix_modules import *


class QMixer(nn.Module):
    def __init__(self, observation_space, model_params):
        super(QMixer, self).__init__()
        if model_params is None:
            model_params = {}
        self.num_agents = len(observation_space)
        self.state_size = sum(np.prod(_.shape) for _ in observation_space.values())
        self.hidden_size = model_params.get("hidden_size", 32)
        # self.state_size = self.hidden_size*4

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
        self.hidden_size_t1 = self.hidden_size_t2 = model_params.get("hidden_size", 64)
        self.coord_recurrent_size = model_params.get("coord_recurrent_size", 256)
        self.shared_comm_ae = model_params.get("ae_comm", False)
        self.cnn_input_proc = model_params.get("cnn_input_proc", False)
        self.eval_coord_mask = model_params.get("eval_coord_mask", False)
        # setup probabilities tensor for training
        self.delayed_comm = False
        self.comm_delay_factors = torch.Tensor([1, 0.75, 0.5, 0.25, 0.125]).to(self._device)
        _skewness = torch.tensor([0.75, 0.15, 0.03, 0.001, 0.00001])
        self.comm_delays_probs = torch.exp(_skewness.log() - torch.max(_skewness.log()))
        self.comm_delays_probs /= torch.sum(self.comm_delays_probs).to(self._device)  # (.8, .16, .03, .001, .00001)

        # setup network modules
        _to_ma = lambda m, args, kwargs: nn.ModuleList([m(*args, **kwargs) for _ in range(self.num_agents)])
        _init_fc = lambda m: init(m, lambda x: nn.init.kaiming_normal_(x, nonlinearity='relu'), lambda x: nn.init.constant_(x, 0))

        # (1) preprocess input to extract relevant features (shared)
        if self.cnn_input_proc:
            self.input_processor = CNNProc(3, self.hidden_size_t1, size=model_params.get("input_proc_size", 0))
        else:
            self.input_processor = FCProc(self.obs_size, self.hidden_size_t1, size=model_params.get("input_proc_size", 0))
        # (2) outgoing communication
        if self.shared_comm_ae:
            self.comm_size = model_params.get("ae_comm_size", 16)
            self.ae = EncoderDecoder(self.hidden_size_t1+self.action_size, self.comm_size)
        else:
            # in theory the msg is formed as concatenation of observation and action
            # i.e. self.comm_size = self.obs_size+self.action_size
            # in practice the message sent is already the final processed_input+actions for sake of computational effort
            self.comm_size = self.hidden_size_t1+self.action_size
        # (3) incoming communication
        self._ids_one_hot = torch.eye(self.num_agents).to(self._device)
        self.plan_size = self.num_agents+self.comm_size  # agent one_hot identifier + hidden + action one_hot

        # Q network to take decisions independently of others
        self.ma_q = QNet(self.agents_ids, self.action_space, self.hidden_size_t1)
        # Produce the coordination mask
        self.ma_coordinator = Coordinator(self.agents_ids, self.action_space, plan_size=self.plan_size, coord_recurrent_size=self.coord_recurrent_size)
        # Coordinated Q network to `slightly` adjust your decisions
        self.intent_extractor = nn.Sequential(_init_fc(nn.Linear(self.plan_size, self.hidden_size_t2)),
                                              nn.ReLU(),
                                              _init_fc(nn.Linear(self.hidden_size_t2, self.hidden_size_t2)))
        self.co_q_linear = nn.ModuleList([
            nn.Sequential(_init_fc(nn.Linear(self.hidden_size_t1+self.hidden_size_t2, self.hidden_size_t2)),
                          nn.ReLU(),
                          _init_fc(nn.Linear(self.hidden_size_t2, action_space[id].n)),
                          nn.Sigmoid()
            ) for id in agents_ids])

    def get_coordinator_parameters(self):
        coordinator_net_params = []
        coordinator_net_params += self.ma_coordinator.parameters()

        return coordinator_net_params

    def get_policy_parameters(self):
        policy_net_params = []
        policy_net_params += self.input_processor.parameters()
        policy_net_params += self.ma_q.parameters()
        policy_net_params += self.intent_extractor.parameters()
        # policy_net_params += self.co_q_net.parameters()
        policy_net_params += self.co_q_linear.parameters()

        return policy_net_params

    def to(self, device):
        self._device = device
        super().to(device)
        # ...the standard recursion applied to submodules set only the modules parameters and not the `_device` variable
        self.comm_delay_factors = self.comm_delay_factors.to(device)
        self.comm_delays_probs = self.comm_delays_probs.to(device)
        self._ids_one_hot = self._ids_one_hot.to(device)
        self.ma_q.to(device)
        self.ma_coordinator.to(device)
        return self

    def init_hidden(self, batch_size=1):
        return self.ma_q.init_hidden(batch_size), self.ma_coordinator.init_hidden(batch_size)

    def _encode_msgs(self, x, solo_actions_one_hot, train_ae=False):
        ae_loss = 0
        if self.shared_comm_ae:
            current_plans = []
            plans = torch.cat([self._ids_one_hot.repeat(x.size(0), 1, 1), x, solo_actions_one_hot], dim=-1)
            for i in range(self.num_agents):
                enc_x, dec_x = self.ae(plans[:, i])
                if train_ae: ae_loss += F.mse_loss(dec_x, plans[:, i])
                current_plans.append((dec_x if self.training else enc_x).unsqueeze(1))  # (*)
            current_plans = torch.cat(current_plans, dim=1).detach()
        else:
            current_plans = torch.cat([self._ids_one_hot.repeat(x.size(0), 1, 1), x, solo_actions_one_hot], dim=-1).detach()  # (**)
        return current_plans, ae_loss

    def _decode_msgs(self, comm):
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

    # def _modify_qs(self, action_logits, hiddens, comm_plans, masks):
    #     # Compute q modifications on the base of the coordination induced by the coordination boolean mask
    #     # `action_logits` (req_grad), `masks` (detached), `comm_msgs` (detached)
    #     q_values = []
    #     for i, id in enumerate(self.agents_ids):
    #         rnn_hxs = hiddens[:, i].clone()
    #         for j in range(self.num_agents):
    #             if i == j: continue
    #             comm_plans_masked = comm_plans[:, j] * masks[i, j]
    #             batch_mask = torch.any(comm_plans_masked, -1).unsqueeze(-1)
    #             batch_comm_plans_masked = torch.masked_select(comm_plans_masked, batch_mask).reshape(-1, comm_plans_masked.size(-1))
    #             batch_rnn_hxs = torch.masked_select(rnn_hxs, batch_mask).reshape(-1, rnn_hxs.size(-1))
    #             if len(batch_comm_plans_masked) > 0:  # certain versions of PyTorch don't like empty batches
    #                 batch_rnn_hxs = self.co_q_net[i](batch_comm_plans_masked, batch_rnn_hxs)
    #                 rnn_hxs = rnn_hxs.masked_scatter(batch_mask, batch_rnn_hxs)
    #         action_logits_coord = action_logits[:, i].clone()
    #         if not torch.equal(rnn_hxs, hiddens[:, i]):
    #             action_logits_coord[torch.any((rnn_hxs-hiddens[:, i]).detach(),-1)] += self.co_q_linear[i](rnn_hxs[torch.any((rnn_hxs-hiddens[:, i]).detach(),-1)])
    #         q_values.append(action_logits_coord.unsqueeze(1))  # NOTE to make it work my hidden vector should be enough informative to allow inference of solo action_logitsand so modify this previous output
    #     q_values = torch.cat(q_values, dim=1)
    # 
    #     return q_values

    # def _modify_qs2(self, action_logits, hiddens, comm_plans, masks, temporal_delays=None):
    #     # Compute q modifications on the base of the coordination induced by the coordination boolean mask
    #     # `action_logits` (req_grad), `masks` (detached), `comm_msgs` (detached)
    #     b,n,h = comm_plans.shape
    #     comm = comm_plans.unsqueeze(1).expand((b, n, n, h))
    #     intents = torch.zeros(b * n * n, self.hidden_size_t2, device=self._device)
    #     masks = masks.permute((2, 0, 1, 3)) * (~torch.eye(n, dtype=torch.bool, device=self._device)).repeat(b, 1, 1).unsqueeze(-1)
    #     comm = (comm * masks).reshape(b * n * n, -1)
    #     intents[comm.any(-1)] = self.intent_extractor(comm[comm.any(-1)])
    #     intents = intents.reshape(b, n, n, -1) / (temporal_delays if temporal_delays is not None else 1)
    #     intents = intents.sum(-2)
    # 
    #     q_values = []
    #     for i, id in enumerate(self.agents_ids):
    #         action_logits_coord = action_logits[:, i].clone()
    #         if torch.any(intents[:, i]):
    #             rnn_hxs = hiddens[:, i].clone()
    #             action_logits_coord *= 1+self.co_q_linear[i](torch.cat([rnn_hxs, intents[:, i]], -1))
    #         q_values.append(action_logits_coord.unsqueeze(1))
    #     q_values = torch.cat(q_values, dim=1)
    # 
    #     return q_values

    def _coordinate_intentions(self, action_logits, hiddens, intents, coord_masks, dones_mask, train_coord=False):
        # Compute q modifications on the base of the coordination induced by the coordination boolean mask
        # `action_logits` (req_grad), `masks` (detached), `comm_msgs` (detached)
        def update_q(masks):
            # apply coordination mask to intents
            bs, n = intents.shape[:2]
            masked_intents = masks.permute((2, 0, 1, 3)) * (~torch.eye(n, dtype=torch.bool, device=self._device)).repeat(bs, 1, 1).unsqueeze(-1) * intents.clone()

            q_values = []
            for i in range(self.num_agents):
                action_logits_coord = action_logits[:, i].clone()
                # sum over agents dimension
                batch_mask = masked_intents[:, i].any(-1).sum(-1)>0
                if torch.any(batch_mask):
                    rnn_hxs = hiddens[:, i].clone()
                    masked_intents_a = masked_intents[:, i][batch_mask].sum(-2) / masked_intents[:, i][batch_mask].any(-1).sum(-1).unsqueeze(-1)  # zero div if no comm
                    action_logits_coord[batch_mask] = action_logits_coord[batch_mask] * \
                        (1+self.co_q_linear[i](torch.cat([rnn_hxs[batch_mask], masked_intents_a], -1)))
                q_values.append(action_logits_coord.unsqueeze(1))
            q_values = torch.cat(q_values, dim=1)
            return q_values

        q_values = update_q(coord_masks)
        if train_coord:
            done_matrix = dones_mask.permute(1, 2, 0) * dones_mask.permute(2, 1, 0)
            with torch.no_grad():
                if self.eval_coord_mask == "true":
                    inverse_q_values = update_q((torch.ones_like(coord_masks) * done_matrix.unsqueeze(-1)).bool())
                elif self.eval_coord_mask == "inverse":
                    inverse_q_values = update_q((~coord_masks * done_matrix.unsqueeze(-1)).bool())
                elif self.eval_coord_mask == "optout":
                    inverse_q_values = []
                    for i in range(self.num_agents):
                        mask = coord_masks.clone()
                        mask[:, i] = ~mask[:, i]
                        inverse_q_values.append(update_q((mask * done_matrix.unsqueeze(-1)).bool()).unsqueeze(2))
                    inverse_q_values = torch.cat(inverse_q_values, dim=2)
        else:
            inverse_q_values = None

        return q_values, inverse_q_values

    def forward(self, state, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans=None, coord_masks=None, rec_intents=None, comm_ts_delays=None, train_coord=False, train_ae=False):
        assert not train_coord or train_coord and (comm_plans is None and coord_masks is None), "The arguments combination passed do not match a std wanted behaviour."
        assert not (comm_plans is None and coord_masks is None) or (comm_plans is None and coord_masks is None), "The arguments combination passed is not valid."

        # shared input processing
        bs, n = state.shape[:2]
        if self.cnn_input_proc:
            input = state.reshape(bs * n, *self.obs_shape).transpose(-1, -3)
        else:
            input = state.reshape(bs * n, -1)
        x = self.input_processor(input)
        x = x.reshape(bs, n, -1)

        # --- (1) Solo action ---
        solo_qs, rnn_hxs = self.ma_q(x, rnn_hxs)
        solo_actions = torch.argmax(solo_qs, dim=-1)
        solo_actions_one_hot = torch.zeros_like(solo_qs).scatter_(-1, solo_actions.type(torch.int64).unsqueeze(-1), 1.)

        # --- (2) Communication ---
        current_plans, ae_loss = self._encode_msgs(x, solo_actions_one_hot, train_ae)

        # --- (3) Coordination ---
        # retrieve incoming messages: are the current timestep communicated plans
        comm_msgs = current_plans

        # process incoming messages from others. Also, they need to be masked with dones
        proc_comm = (self._decode_msgs(comm_msgs) if comm_plans is None else comm_plans) * dones_mask

        # produce mask of coordination using incoming messages
        if coord_masks is None:
            coord_masks, glob_rnn_hxs = self.ma_coordinator(current_plans, proc_comm, glob_rnn_hxs)
            done_matrix = dones_mask.permute(1, 2, 0) * dones_mask.permute(2, 1, 0)
            if self.training:  # mask here is for `coord_masks` output variable
                coord_masks = F.gumbel_softmax(coord_masks, hard=True, dim=-1) * done_matrix.unsqueeze(-1)  # add randomness proportional to logits relative value
            blind_coord_masks = (torch.argmax(coord_masks, -1, keepdim=True) * done_matrix.unsqueeze(-1)).bool().detach()  # argmax into bool: 0=no coord, 1=coord
        else:
            blind_coord_masks = coord_masks.permute(1,2,0,3)

        # extract intentions from messages and following the coordination mask use them to update the q values
        intents = self.intent_extractor(proc_comm.unsqueeze(1).repeat((1, n, 1, 1)))
        if rec_intents is not None:
            intents[comm_ts_delays>0] = rec_intents[comm_ts_delays>0]
        temporal_delays = 1 if comm_ts_delays is None else torch.gather(self.comm_delay_factors, 0, torch.clamp(comm_ts_delays, 0, len(self.comm_delay_factors)-1).long().flatten()).view(bs, n, n, 1).to(self._device)
        scaled_intents = intents * temporal_delays
        qs, inv_qs = self._coordinate_intentions(solo_qs, rnn_hxs, scaled_intents, blind_coord_masks, dones_mask, train_coord)

        # actions from done agents are not useful in this implementation, so are dropped in the output
        return qs, rnn_hxs, glob_rnn_hxs, inv_qs, coord_masks, (proc_comm, blind_coord_masks.permute(2,0,1,3), intents), ae_loss

    def sample_action_from_qs(self, qs):
        """Compute the probability distribution from Q values and sample to obtain the action."""
        # pack batch and agents and sample the distributions
        batch_size, n_agents, _ = qs.shape
        action = torch.multinomial(torch.softmax(qs, dim=-1).view(batch_size*n_agents, -1), 1).view(batch_size, n_agents)
        return action

    def take_action(self, state, rnn_hxs, glob_rnn_hxs, dones_mask, prev_intents=None, comm_delays=None):
        """Predict Qs and from those sample an action."""
        qs, rnn_hxs, glob_rnn_hxs, _, _, additional_input, _ = self.forward(state, rnn_hxs, glob_rnn_hxs, dones_mask, rec_intents=prev_intents, comm_ts_delays=comm_delays)
        # sample action to use in the env from q
        action = self.sample_action_from_qs(qs)
        intents = additional_input[-1]
        return action.detach(), rnn_hxs.detach(), glob_rnn_hxs.detach(), intents.detach()

    def eval_action(self, state, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans, coord_masks, actions):
        """Off policy call returning Q of given actions."""
        bs, n = state.shape[:2]
        delays = None if not self.delayed_comm else torch.multinomial(self.comm_delays_probs, num_samples=bs * n * n, replacement=True)
        qs, rnn_hxs, glob_rnn_hxs, _, _, _, _ = self.forward(state, rnn_hxs, glob_rnn_hxs, dones_mask, comm_plans, coord_masks, comm_ts_delays=delays)
        # gather q of action passed
        q_a = qs.gather(-1, actions.long()).squeeze(-1)
        return q_a, rnn_hxs, glob_rnn_hxs
