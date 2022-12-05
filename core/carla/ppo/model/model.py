import math

import torch
import torch.nn as nn
from gym.spaces import Dict

from core.carla.ppo.model.distributions import Categorical, DiagGaussian
from core.carla.ppo.model.utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    """
    Directly predict n values to sample from a learnt distribution: Categorical if Discrete action space else Gaussian
    The output size is dependent to the action size of the environment
    """
    def __init__(self, obs_space, action_space, recurrent, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.action_space = list(action_space.values())[0]
        self.obs_space = list(obs_space.values())[0]
        self._recurrent = recurrent

        # handle implementation with or without vocabulary of measurements
        if isinstance(self.obs_space, Dict):
            img_num_features = self.obs_space.spaces['img'].shape[0]
            msr_num_features = self.obs_space.spaces['msr'].shape[0]
        else:
            img_num_features = self.obs_space.shape[0]
            msr_num_features = None

        if recurrent:
            self.base = CNNGRU(img_num_features, msr_num_features, **base_kwargs)
        else:
            self.base = CNNBase(img_num_features, msr_num_features, **base_kwargs)

        # setup distribution where to sample actions
        if self.action_space.__class__.__name__ == "Discrete":  # discrete
            num_outputs = self.action_space.n
            self.a_dist = Categorical(self.base.output_size, num_outputs)
        elif self.action_space.__class__.__name__ == "Box":  # continuous
            num_outputs = self.action_space.shape[0]
            self.a_dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
        # setup distribution where to sample communication intentions
        self.c_dist = Categorical(self.base.output_size, 2)

    @property
    def is_recurrent(self):
        return self._recurrent

    def forward(self, input, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, img, msr, rnn_hxs, comm, masks, eps, deterministic=False):
        value, actor_features, rnn_hxs = self.base(img, msr, rnn_hxs, comm, masks)
        a_dist = self.a_dist(actor_features)
        c_dist = self.c_dist(actor_features)

        if torch.rand(1).item() < eps:  # act at random if lower than epsilon
            action = torch.tensor([[self.action_space.sample()]]).type_as(actor_features)
            comm_mask = torch.randn((actor_features.size(0),1)).type_as(actor_features)
        elif deterministic:  # maximum probability
            action = a_dist.mode()
            comm_mask = c_dist.mode()
        else:  # sample from probability distribution
            action = a_dist.sample()
            comm_mask = c_dist.sample()
        # cat the probabilities to hide complexity
        log_probs = torch.cat([a_dist.log_probs(action), c_dist.log_probs(comm_mask)], 1)
        output = torch.cat([action, comm_mask], 1)

        return output, log_probs, value, rnn_hxs

    def get_value(self, img, msr, rnn_hxs, comm, masks):
        return self.base(img, msr, rnn_hxs, comm, masks)[0]

    def evaluate_actions(self, img, msr, rnn_hxs, comm, masks, output):
        value, actor_features, rnn_hxs = self.base(img, msr, rnn_hxs, comm, masks)
        a_dist = self.a_dist(actor_features)
        c_dist = self.c_dist(actor_features)
        # action and communication mask are handled together to hide complexity
        actions, comm_mask = output.split(output.size(1)-1, 1)
        log_probs = torch.cat([a_dist.log_probs(actions), c_dist.log_probs(comm_mask)], 1)
        dist_entropy = a_dist.entropy().mean() + c_dist.entropy().mean()

        return log_probs, value, dist_entropy, rnn_hxs


class CNNBase(nn.Module):
    def __init__(self, img_num_features, msr_num_features, hidden_size=512):
        super(CNNBase, self).__init__()

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

        self.init_fc = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        if msr_num_features != 0:
            self.fc_enc = nn.Sequential(
                self.init_fc(nn.Linear(msr_num_features, hidden_size//4)),
                nn.ReLU(),
                self.init_fc(nn.Linear(hidden_size//4, hidden_size//2)),
                nn.ReLU(),
                self.init_fc(nn.Linear(hidden_size//2, hidden_size)),
                nn.ReLU()
            )

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

    @property
    def output_size(self):
        return self._hidden_size

    def _backbone_proc(self, img, msr, comm):
        x_img = self.cnn_backbone(img)

        x_msr = self.fc_enc(msr)
        # x_comm = self.comm_enc(comm)
        # x = torch.cat([x_img, x_msr, comm], -1)
        # x = self.fc_joint(x)
        x = torch.cat([x_img, x_msr, comm], -1)
        x = self.fc_joint(x)
        # else:
        #     x = x_img
        return x

    def forward(self, img, msr, rnn_hxs=None, masks=None):
        x = self._backbone_proc(img, msr)

        return self.critic_linear(x), x, rnn_hxs

class CNNGRU(CNNBase):
    def __init__(self, img_num_features, msr_num_features, hidden_size, recurrent_input_size):
        super(CNNGRU, self).__init__(img_num_features, msr_num_features, hidden_size)

        self._recurrent_input_size = recurrent_input_size

        self.gru = nn.GRU(hidden_size, recurrent_input_size)  # input features, recurrent steps
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # self.comm_encoder = nn.Linear(recurrent_input_size, recurrent_input_size)

        self.comm_enc = nn.Sequential(
            self.init_fc(nn.Linear(recurrent_input_size, hidden_size // 4)),
            nn.ReLU(),
            self.init_fc(nn.Linear(hidden_size // 4, hidden_size // 2)),
            nn.ReLU(),
            self.init_fc(nn.Linear(hidden_size // 2, recurrent_input_size)),
            nn.ReLU()
        )

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

    def forward(self, img, msr, rnn_hxs, comm, done_masks):
        x = super()._backbone_proc(img, msr, comm)

        # x_comm = self.comm_enc(comm)
        # x = x + x_comm

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

        return self.critic_linear(x), x, rnn_hxs
