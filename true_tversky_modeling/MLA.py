import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from einops import rearrange

from .TverskyLayer import TverskyLayer
from .zRMSNorm import ZeroCenteredRMSNorm


# Largely from https://github.com/fla-org/flash-linear-attention
# But without rope and support tversky.

class VarlenMLA(nn.Module):
    def __init__(self, config, layer_idx=0, tversky=False, shared_tversky_features=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_attn_heads
        self.layer_idx = layer_idx

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_down = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(self.q_lora_rank)
        self.q_up = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        self.kv_down = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_norm = ZeroCenteredRMSNorm(self.kv_lora_rank)
        self.kv_up = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_head_dim + self.v_head_dim), bias=False)

        # [ln(x + 1) / 2] + 1 -> ~[1, 2.5] scale factor for 0-23
        scale_factor = ((torch.log(torch.tensor(layer_idx + 1, dtype=torch.float)) / 2) + 1).item()

        if tversky:
            self.w_o = TverskyLayer(
                input_dim=self.n_heads * self.v_head_dim,
                prototypes=self.hidden_size,
                features=shared_tversky_features,
                prototype_init=lambda x: torch.nn.init.uniform_(x, -0.12 * scale_factor, 0.12 * scale_factor),
            )
        else:
            self.w_o = nn.Linear(self.n_heads * self.v_head_dim, self.hidden_size, bias=False)
            nn.init.normal_(self.w_o.weight, mean=0.0, std=0.004 * scale_factor)

        self.scaling = self.qk_head_dim ** -0.5

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.q_down.weight, mean=0.0, std=0.006)
        nn.init.normal_(self.q_up.weight, mean=0.0, std=0.006)
        nn.init.normal_(self.kv_down.weight, mean=0.0, std=0.006)
        nn.init.normal_(self.kv_up.weight, mean=0.0, std=0.006)

    def forward(self, x, cu_seqlens=None, max_seqlen=None):
        q = self.q_down(x)
        q = self.q_norm(q)
        q = self.q_up(q)

        kv = self.kv_down(x)
        kv = self.kv_norm(kv)
        kv = self.kv_up(kv)

        q = rearrange(q, 's (h d) -> s h d', h=self.n_heads, d=self.qk_head_dim)
        kv = rearrange(kv, 's (h d) -> s h d', h=self.n_heads)

        k, v = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)

        q = q * self.scaling

        # Pad for flash attn varlen
        if self.qk_head_dim != self.v_head_dim:
            v = F.pad(v, [0, self.qk_head_dim - self.v_head_dim])

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True
        )

        # Unpad
        if self.qk_head_dim != self.v_head_dim:
            out = out[:, :, :self.v_head_dim]

        out = rearrange(out, 's h d -> s (h d)')

        return self.w_o(out)
