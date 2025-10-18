import torch
import torch.nn as nn
import torch.nn.functional as F

class Xorz(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim):
        super().__init__()
        self.features = nn.Parameter(torch.empty(feature_dim, input_dim))
        self.output_features = nn.Parameter(torch.empty(output_dim, feature_dim))

        unit_scale_f = (3.0 / input_dim) ** 0.5
        unit_scale_o = (3.0 / num_features) ** 0.5
        nn.init.uniform_(self.features, -unit_scale_f, unit_scale_f)
        nn.init.uniform_(self.output_features, -unit_scale_o, unit_scale_o)

    def forward(self, x):
        A = x @ self.features.T
        sigma = torch.sigmoid(2 * A)
        combined = sigma * (A + 2) - 1
        return combined @ self.output_features.T
