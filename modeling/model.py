import torch
import torch.nn as nn
from dataclasses import dataclass

from .decoder import DecoderLayer

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 512
    intermediate_size_up: int = 1024
    intermediate_size_down: int = 256+128
    attn_shared_proj_size: int = 1536
    num_decoders: int = 24
    n_attn_heads: int = 16

class CoolLanguageModelWowExclamationMark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i, variant=True) for i in range(config.num_decoders)
        ])

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
