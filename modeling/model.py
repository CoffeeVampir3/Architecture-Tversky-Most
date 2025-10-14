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

    # MLA numbers loosely proposed by paper: https://arxiv.org/abs/2506.09342
    q_lora_rank = 384      # 0.75 × hidden
    kv_lora_rank = 256     # 0.5 × hidden
    qk_head_dim = 48       # 1.5 × standard head_dim
    v_head_dim = 32        # 1.0 × standard head_dim

# A block of N Tversky decoders that share the same features.
class SharedTverskyBlock(nn.ModuleList):
    def __init__(self, config, starting_layer_idx, number_decoders):
        super().__init__()
        scale_factor = ((torch.log(torch.tensor(starting_layer_idx + 1, dtype=torch.float)) / 2) + 1).item()

        self.shared_features = nn.Parameter(
            torch.zeros(config.tversky_num_features, config.hidden_size)
        )
        torch.nn.init.uniform_(self.shared_features, -0.1 * scale_factor, 0.1 * scale_factor)
        layers = [
            DecoderLayer(
                config,
                layer_idx=starting_layer_idx + i,
                tversky=True,
                block_shared_tversky_features=self.shared_features
            )
            for i in range(number_decoders)
        ]

        self.extend(layers)

class CoolLanguageModelWowExclamationMark(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList()

        # 10 Regular decoders.
        idx = 0
        num_initial_decoders = 10
        for i in range(num_initial_decoders):
            self.layers.append(
                DecoderLayer(config, layer_idx=i, tversky=False)
            )
        idx += num_initial_decoders

        # self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.output_layer.weight = self.embedding.weight
        # self.reset_parameters()
        # return

        # I'm going to answer the question of why choose this structure:
        # Tversky-All setup using all o-projections and the output head which all shared features struck me as asking
        # far too much of the earliest layers, as those layers much align the projections immediately into the tversky
        # shared feature space. There's relatively little the network can do to form supporting blocks in order to
        # set up the projection.

        # This is the same issue if we ditch the head, we still ask the first layer to do too much in my opinion.
        # So, what if we let regular linears handle figuring out how to support the initial projection into the
        # "tversky feature space"

        # From here, we could probably be fine with having a single block of 9 shared tversky layers and have a
        # roughly similar network. My justification was based on how neocortical columns form repeating, mostly-local
        # dense columns that should act as "somewhat" independent processors. This is obviously not a literal mirroring
        # of the neocortex, but was the inspiration for breaking the features apart like this. Essentialy, my thought is
        # we get to now model local features strongly, instead of forcing the network to model global features weakly.
        # This actually becomes a very useful construct, where we can force feature-modeling in specific places and with
        # specific structure. We can even leak the features to later parts of the model, though I couldn't rationalize
        # how doing that here would make sense. (I didn't test it either.)
        # (When I say global here, I'm not talking about the modeling distribution, but rather the literal constraint
        # that the feature weights in tversky-all are tied together globally, where as here we form "local" subunits" as in
        # those features are only locally shared between a few processors.)

        # Spitballing on the local/global front, but we can potentially arrange combinations of feature and prototype sharing
        # to force complex joint modeling in certain regions and also use it as a method of information-sharing in the network.
        # This would align to a more true-to-neocortical structure that has local/global mixtures of dense interconnections
        # though of course, we don't have recurrence here.

        # Create 3 blocks of Tversky layers with shared features per block. (So each block of 3 has it's own feature bank)
        num_tversky_blocks = 3
        decoders_per_block = 3
        self.tversky_blocks = nn.ModuleList()

        for block_idx in range(num_tversky_blocks):
            block = SharedTverskyBlock(
                config,
                starting_layer_idx=idx,
                number_decoders=decoders_per_block
            )
            self.tversky_blocks.append(block)
            self.layers.extend(block)
            idx += decoders_per_block

        # Rest of the blocks regular decoders.
        for i in range(idx, config.num_decoders):
            self.layers.append(
                DecoderLayer(config, layer_idx=i, tversky=False)
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
