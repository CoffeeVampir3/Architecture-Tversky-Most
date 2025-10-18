import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from einops import rearrange

class VarlenMHA(nn.Module):
    def __init__(self, config, layer_idx=0, **kwargs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_attn_heads

        assert self.hidden_size % self.n_heads == 0
        self.scale_factor = ((torch.log(torch.tensor(layer_idx + 1, dtype=torch.float)) / 2) + 1).item()

        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.gate_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate_proj.weight)
        unit_scale = (3.0 / self.hidden_size) ** 0.5

        # [ln(x + 1) / 2] + 1 -> ~[1, 2.5] scale factor for 0-23
        full_scale = unit_scale * self.scale_factor
        nn.init.uniform_(self.w_o.weight, -full_scale, full_scale)
        nn.init.uniform_(self.w_q.weight, -full_scale, full_scale)
        nn.init.uniform_(self.w_k.weight, -full_scale, full_scale)
        nn.init.uniform_(self.w_v.weight, -full_scale, full_scale)

    def forward(self, x, cu_seqlens=None, max_seqlen=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Where s -> total sequence length (of all batches combined), h -> number attn heads, d -> hidden dimension
        q = rearrange(q, 's (h d) -> s h d', h=self.n_heads)
        k = rearrange(k, 's (h d) -> s h d', h=self.n_heads)
        v = rearrange(v, 's (h d) -> s h d', h=self.n_heads)

        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True
        )

        # g1 headwise gate https://arxiv.org/pdf/2505.06708
        gate_scores = torch.sigmoid(self.gate_proj(x))
        gate_scores = gate_scores.unsqueeze(-1)
        out = out * gate_scores

        out = rearrange(out, 's h d -> s (h d)')
        return self.w_o(out)
