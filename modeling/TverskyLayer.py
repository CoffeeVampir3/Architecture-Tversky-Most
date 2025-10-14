import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLayer(nn.Module):
    def __init__(self,
        input_dim: int,
        prototypes: int | nn.Parameter,
        features: int | nn.Parameter,
        prototype_init_scale=None,
        feature_init_scale=None,
        approximate_sharpness=13
    ):
        super().__init__()

        if isinstance(features, int):
            self.features = nn.Parameter(torch.empty(features, input_dim))
        elif isinstance(features, nn.Parameter):
            self.features = features
        else:
            raise ValueError("features must be int or nn.Parameter")

        if isinstance(prototypes, int):
            self.prototypes = nn.Parameter(torch.empty(prototypes, input_dim))
        elif isinstance(prototypes, nn.Parameter):
            self.prototypes = prototypes
        else:
            raise ValueError("prototypes must be int or nn.Parameter")

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.theta = nn.Parameter(torch.zeros(1))

        self.prototype_init_scale = prototype_init_scale
        self.feature_init_scale = feature_init_scale
        self.approximate_sharpness = approximate_sharpness

        self.reset_parameters()

    def reset_parameters(self):
        if self.prototype_init_scale:
            torch.nn.init.uniform_(self.prototypes, -0.12 * self.prototype_init_scale, 0.12 * self.prototype_init_scale)
        if self.feature_init_scale:
            torch.nn.init.uniform_(self.features, -0.04 * self.feature_init_scale, 0.04 * self.feature_init_scale)

    # Ignorematch with Product intersections
    def forward(self, x):
        B = x.size(0)

        # [B,d] @ [d,F] = [B,F] and [P,d] @ [d,F] = [P,F]
        A = x @ self.features.T          # [B, F]
        Pi = self.prototypes @ self.features.T  # [P, F]

        sigma_A = (torch.tanh(self.approximate_sharpness * A) + 1) * 0.5
        weighted_A = A * sigma_A

        sigma_Pi = (torch.tanh(self.approximate_sharpness * Pi) + 1) * 0.5
        weighted_Pi = Pi * sigma_Pi

        return (self.theta * (weighted_A @ weighted_Pi.T)
                + self.alpha * (weighted_A @ sigma_Pi.T)
                + self.beta * (sigma_A @ weighted_Pi.T)
                - self.alpha * weighted_A.sum(dim=-1, keepdim=True)
                - self.beta * weighted_Pi.sum(dim=-1).unsqueeze(0)) # [B, P]
