import torch
import torch.nn as nn

from .MHA import VarlenMHA
from .TverskyMHA import TverskyVarlenMHA
from .Xorz import Xorz
from .zRMSNorm import ZeroCenteredRMSNorm

class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, variant, **kwargs):
        super().__init__()
        self.self_attn = TverskyVarlenMHA(config, layer_idx, **kwargs) if variant else VarlenMHA(config, layer_idx, **kwargs)
        self.mlp = Xorz(
            config,
            layer_idx,
            intermediate_dim=config.intermediate_size_up if layer_idx % 2 == 1 else config.intermediate_size_down
        )
        self.input_layernorm = ZeroCenteredRMSNorm(config.hidden_size)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size)

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
