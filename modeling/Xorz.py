import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Xorz(nn.Module):
    def __init__(self, config, layer_idx, intermediate_dim=None, shared_features=None):
        super().__init__()
        intermediate_size = intermediate_dim
        if shared_features is not None:
            intermediate_size = shared_features.shape[1]
            self.output_features = shared_features
        else:
            self.output_features = nn.Parameter(torch.empty(config.hidden_size, intermediate_size))

        self.features = nn.Parameter(torch.empty(intermediate_size, config.hidden_size))
        self.had_shared_features = shared_features

        # [ln(x + 1) / 2] + 1 -> ~[1, 2.5] scale factor for 0-23
        self.scale_factor = ((torch.log(torch.tensor(layer_idx + 1, dtype=torch.float)) / 2) + 1).item()
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self):
        rot = self.scale_factor * 3.0
        unit_scale_f = (rot / self.features.shape[1]) ** 0.5
        unit_scale_o = (rot / self.output_features.shape[1]) ** 0.5
        full_scale_f = unit_scale_f
        full_scale_o = unit_scale_o
        nn.init.uniform_(self.features, -full_scale_f, full_scale_f)
        if self.had_shared_features is None:
            nn.init.uniform_(self.output_features, -full_scale_o, full_scale_o)

    def forward(self, x):
        A = x @ self.features.T
        sigma = torch.sigmoid(2 * A)
        combined = sigma * (A + 2) - 1
        return self.scale * (combined @ self.output_features.T)
