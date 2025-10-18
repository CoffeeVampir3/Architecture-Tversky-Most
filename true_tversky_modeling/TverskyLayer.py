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
        approximate_sharpness=3
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
        # Variance vaguely scaled to unit norm scaled by a depth multiplier.
        if self.prototype_init_scale:
            unit_scale_p = (3.0 / self.prototypes.size(1)) ** 0.5
            proto_scale = unit_scale_p * self.prototype_init_scale
            torch.nn.init.uniform_(self.prototypes, -proto_scale, proto_scale)
        if self.feature_init_scale:
            unit_scale_f = (3.0 / self.features.size(1)) ** 0.5
            feature_scale = unit_scale_f * self.feature_init_scale
            torch.nn.init.uniform_(self.features, -feature_scale, feature_scale)

        nn.init.uniform_(self.alpha, -0.05, 0.05)
        nn.init.uniform_(self.beta, -0.05, 0.05)
        nn.init.uniform_(self.theta, -0.1, 0.1)

    def forward(self, x):
        B = x.size(0)

        # somewhere between a norm and a logarithm (defined for +-), but it doesn't generate intermediates
        # so this is substantially more memory efficient (than an RMSnorm e.g.)
        features_norm = torch.asinh(self.features).T
        prototype_norm = torch.asinh(self.prototypes)

        # [B,d] @ [d,F] = [B,F] and [P,d] @ [d,F] = [P,F]
        A = x @ features_norm         # [B, F]
        Pi = prototype_norm @ features_norm  # [P, F]

        weighted_A, sigma_A = self.indicator(A)
        weighted_Pi, sigma_Pi = self.indicator(Pi)

        result = (self.theta * (weighted_A @ weighted_Pi.T)
                + self.alpha * (weighted_A @ sigma_Pi.T)
                + self.beta * (sigma_A @ weighted_Pi.T)
                - self.alpha * weighted_A.sum(dim=-1, keepdim=True)
                - self.beta * weighted_Pi.sum(dim=-1).unsqueeze(0)) # [B, P]

        return result.to(torch.bfloat16)

    def indicator(self, x):
        # Forward
        sigma_hard = (x > 0).to(x.dtype)

        # STE
        sigma_smooth = torch.sigmoid(self.approximate_sharpness * x)
        sigma = sigma_smooth + (sigma_hard - sigma_smooth).detach()

        # Exploit hot cache of sigma and maybe x
        weighted = x * sigma
        return weighted, sigma

    # def forward(self, x):
    #     B = x.size(0)

    #     features_norm = torch.asinh(self.features).T
    #     prototype_norm = torch.asinh(self.prototypes)

    #     # [B,d] @ [d,F] = [B,F] and [P,d] @ [d,F] = [P,F]
    #     A = x @ features_norm        # [B, F]
    #     Pi = prototype_norm @ features_norm  # [P, F]

    #     weighted_A, sigma_A = self.indicator(A)     # both [B, F]
    #     weighted_Pi, sigma_Pi = self.indicator(Pi)  # both [P, F]

    #     common = weighted_A @ weighted_Pi.T            # [B,F] @ [F,P] = [B,P]

    #     distinctive_A = weighted_A @ (1 - sigma_Pi).T  # [B,F] @ [F,P] = [B,P]
    #     distinctive_B = (1 - sigma_A) @ weighted_Pi.T  # [B,F] @ [F,P] = [B,P]

    #     #  S = θ·C - α·D_A - β·D_B
    #     return self.theta * common - self.alpha * distinctive_A - self.beta * distinctive_B # [B, P]

    #Ignorematch with Product intersections
    # def forward(self, x):
    #     B = x.size(0)

    #     # [B,d] @ [d,F] = [B,F] and [P,d] @ [d,F] = [P,F]
    #     A = x @ self.features.T          # [B, F]
    #     Pi = self.prototypes @ self.features.T  # [P, F]

    #     weighted_A, sigma_A = self.indicator(A)
    #     weighted_Pi, sigma_Pi = self.indicator(Pi)

    #     return (self.theta * (weighted_A @ weighted_Pi.T)
    #             + self.alpha * (weighted_A @ sigma_Pi.T)
    #             + self.beta * (sigma_A @ weighted_Pi.T)
    #             - self.alpha * weighted_A.sum(dim=-1, keepdim=True)
    #             - self.beta * weighted_Pi.sum(dim=-1).unsqueeze(0)) # [B, P]

    # def norm(self, x):
    #     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    # def forward(self, x):
    #     B = x.size(0)
    #     P = self.prototypes.size(0)

    #     # somewhere between a norm and a logarithm (defined for +-), but it doesn't generate intermediates
    #     # so this is substantially more memory efficient
    #     features_norm = torch.asinh(self.features).T
    #     prototype_norm = torch.asinh(self.prototypes)
    #     # [B,d] @ [d,F] = [B,F] and [P,d] @ [d,F] = [P,F]
    #     A = x @ features_norm          # [B, F]
    #     Pi = prototype_norm @ features_norm  # [P, F]

    #     weighted_A, sigma_A = self.indicator(A)
    #     weighted_Pi, sigma_Pi = self.indicator(Pi)

    #     # K * (weighted_A @ weighted_Pi.T)
    #     result = torch.addmm(
    #         torch.empty(B, P, device=x.device, dtype=x.dtype),
    #         weighted_A, weighted_Pi.T,
    #         beta=0, alpha=5e-3
    #     )

    #     # - K * (weighted_A @ (1 - sigma_Pi).T)
    #     result = torch.addmm(
    #         result,
    #         weighted_A, (1 - sigma_Pi).T,
    #         beta=1, alpha=-5e-3
    #     )

    #     # - K * ((1 - sigma_A) @ weighted_Pi.T)
    #     result = torch.addmm(
    #         result,
    #         (1 - sigma_A), weighted_Pi.T,
    #         beta=1, alpha=-5e-3
    #     )

    #     return result
