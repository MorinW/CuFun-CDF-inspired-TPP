import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CuFun.tpp.nn import BaseModule


class NonnegativeLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self.weight.data.abs_()

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.weight.data.clamp_(0.0)
        return F.linear(input, self.weight, self.bias)


class FullyNN(BaseModule):
    def __init__(self, config, n_layers=2, layer_size=64):
        super().__init__()
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)

        self.linear_time = NonnegativeLinear(1, layer_size)
        if self.using_history:
            self.linear_rnn = nn.Linear(config.history_size, layer_size, bias=False)
        if self.using_embedding:
            self.linear_emb = nn.Linear(config.embedding_size, layer_size, bias=False)

        self.linear_layers = nn.ModuleList([
            NonnegativeLinear(layer_size, layer_size) for _ in range(n_layers - 1)
        ])
        self.final_layer = NonnegativeLinear(layer_size, 1)
        self.add_layer = NonnegativeLinear(layer_size, layer_size)

    def mlp(self, y, h=None, emb=None):
        y = y.unsqueeze(-1)
        hidden = self.linear_time(y)
        if h is not None:
            tmp = self.linear_rnn(h)
            tmp_mean = torch.mean(tmp)
            hidden_mean = torch.mean(hidden)
            hidden *= tmp
        hidden = torch.tanh(hidden)

        for linear in self.linear_layers:
            hidden = torch.tanh(linear(hidden))
        hidden = self.final_layer(hidden)
        return hidden.squeeze(-1), tmp_mean, hidden_mean

    def log_prob(self, y, h=None, emb=None):
        y.requires_grad_()
        output = self.mlp(y, h, emb)[0]
        comp_term1 = self.mlp(y, h, emb)[1]
        comp_term2 = self.mlp(y, h, emb)[2]
        comp = [comp_term1, comp_term2]
        density_integral = torch.sigmoid(output)
        density = torch.autograd.grad(density_integral, y, torch.ones_like(output), create_graph=True)[0]
        log_p = torch.log(density + 1e-8)
        s = 1-density_integral
        Phi = -torch.log(1-density_integral)
        intensity = density/s
        return log_p, comp
