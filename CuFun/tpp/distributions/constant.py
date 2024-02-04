import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CuFun.tpp.nn import BaseModule


class NonnegativeLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # Make weight non-negative at initialization
        self.weight.data.abs_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.weight.data.clamp_(0.0)
        return F.linear(input, self.weight, self.bias)


class Const(BaseModule):
    def __init__(self, config, n_layers=2, layer_size=64):
        super().__init__()
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)

        self.linear_time = NonnegativeLinear(1, layer_size)
        if self.using_history:
            self.linear_rnn = nn.Linear(config.history_size, 1, bias=False)
        if self.using_embedding:
            self.linear_emb = nn.Linear(config.embedding_size, 1, bias=False)

        self.linear_layers = nn.ModuleList([
            NonnegativeLinear(layer_size, layer_size) for _ in range(n_layers - 1)
        ])
        self.final_layer = NonnegativeLinear(layer_size, 1)

    def cdf(self, y, h=None, emb=None):
        output = self.mlp(y, h, emb)
        integral = F.softplus(output)
        return -torch.expm1(-integral)

    def log_cdf(self, y, h=None, emb=None):
        return torch.log(self.cdf(y, h, emb) + 1e-8)

    def sigmoid(self, input):
        output = 1 / (1 + torch.exp(-4 * input - 18))
        return output

    def constrain(self, input, y):
        y = torch.clamp(y, 0, 1)
        output = y / (1 + torch.exp(-input))
        return output

    def softplus(self, input, k):
        output = torch.log(1+k*torch.exp(input))/k
        return output

    def log_prob(self, y, h=None, emb=None):
        y.requires_grad_()
        y = y.unsqueeze(-1)
        log_intensity = self.linear_rnn(h)
        integral = torch.exp(log_intensity) * y
        log_p = (log_intensity - integral).squeeze(-1)
        s = torch.exp(log_p) / torch.exp(log_intensity).squeeze(-1)
        return log_p, s

