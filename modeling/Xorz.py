import torch
import torch.nn as nn
import torch.nn.functional as F

class Xorz(nn.Module):
    def __init__(self, scale_factor, input_dim, output_dim, feature_dim):
        super().__init__()
        self.features = nn.Parameter(torch.empty(feature_dim, input_dim))
        self.output_features = nn.Parameter(torch.empty(output_dim, feature_dim))
        self.scale_factor = scale_factor
        self.reset_parameters()

    def reset_parameters(self):
        rot = self.scale_factor * 3.0
        unit_scale_f = (rot / self.features.shape[1]) ** 0.5
        unit_scale_o = (rot / self.output_features.shape[1]) ** 0.5
        full_scale_f = unit_scale_f
        full_scale_o = unit_scale_o
        nn.init.uniform_(self.features, -full_scale_f, full_scale_f)
        nn.init.uniform_(self.output_features, -full_scale_o, full_scale_o)

    def forward(self, x):
        A = x @ self.features.T
        sigma = torch.sigmoid(2 * A)
        combined = sigma * (A + 2) - 1
        return combined @ self.output_features.T
