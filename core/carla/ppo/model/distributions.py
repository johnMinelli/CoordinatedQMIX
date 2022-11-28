import torch
import torch.nn as nn

from core.carla.ppo.model.utils import AddBias, init


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init. constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

        # self.fc_mean = nn.Linear(num_inputs, num_outputs)
        # self.fc_mean.bias.data[0] = 0.5  # Throttle and Brake
        # self.fc_mean.bias.data[1] = 0.0  # Steer
        # self.fc_mean.weight.data.fill_(0)
        # self.logstd = AddBias(-2.3 * torch.ones(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())





def select_action(args, action_out):
    if self.continuous:
        action_mean = self.action_mean(next_hid)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        action_out, _ , _ = (action_mean, action_log_std, action_std), v, ret
    else:
        action_out, _ , _ = [F.log_softmax(head(next_hid), dim=-1) for head in self.heads], v, ret

    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
        return ret