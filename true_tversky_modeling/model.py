import torch
import torch.nn as nn
from dataclasses import dataclass

from .decoder import DecoderLayer
from .TverskyLayer import TverskyLayer

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_decoders: int = 24
    n_attn_heads: int = 16
    tversky_num_features: int = 2048
    tversky_attn_feature_size: int = 512

class CoolLanguageModelWowExclamationMark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList()

        tversky_groups = [(10, 13), (13, 16), (16, 19)]

        shared_features = {}
        for start, end in tversky_groups:
            scale_factor = ((torch.log(torch.tensor(start + 1, dtype=torch.float)) / 2) + 1).item()
            unit_scale = (3.0 / config.hidden_size) ** 0.5
            feature_scale = unit_scale * scale_factor

            features = nn.Parameter(torch.empty(config.tversky_num_features, config.hidden_size))
            nn.init.uniform_(features, -feature_scale, feature_scale)
            shared_features[(start, end)] = features

        for i in range(config.num_decoders):
            is_tversky = False
            block_features = None

            for start, end in tversky_groups:
                if start <= i < end:
                    is_tversky = True
                    block_features = shared_features[(start, end)]
                    break

            self.layers.append(
                DecoderLayer(config, layer_idx=i, tversky=is_tversky,
                           block_shared_tversky_features=block_features)
            )

        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def get_embeddings(self, input_ids, cu_seqlens=None, max_seqlen=None):
        """
        Returns embeddings before classifier head.
        input_ids: [total_seq_len] for varlen
        cu_seqlens, max_seqlen: for flash attention layers
        Returns: [total_seq_len, embed_size]
        """
        x = self.embedding(input_ids)  # [total_seq_len, embed_size]

        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        return x  # [total_seq_len, embed_size]

    def get_classifier_weights(self):
        return self.output_layer.weight

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None):
        """
        input_ids: [total_seq_len]
        Returns: [total_seq_len, vocab_size]
        """
        embeddings = self.get_embeddings(input_ids, cu_seqlens, max_seqlen)
        logits = self.output_layer(embeddings)
        return logits
