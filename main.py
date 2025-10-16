import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.amp import autocast
from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.utils import compute_z_loss
from safetensors.torch import save_model, load_model
from muon import SingleDeviceNorMuonWithAuxAdam

from utils.jsonl_dataloader import JSONLDirectoryTokenizedDataset, varlen_collate_fn
from utils.trainutils import count_parameters_layerwise, TBLogger
from modeling.model import CoolLanguageModelWowExclamationMark, ModelConfig
from modeling.zRMSNorm import ZeroCenteredRMSNorm
from modeling.TverskyLayer import TverskyLayer

import os
import torch._inductor.config

# There's a memory leak related to very specific interactions of AMP + flash attn varlen + Tversky O projections.
# The exact nature is unclear to me but these settings (specifically the dyanic graph one) prevents the leak.
# The expandable segments seems to reduce the overall allocation size
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

#torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')

def compute_varlen_loss_with_cce(model, concatenated_input_ids, cu_seqlens, sequence_lengths):
    device = concatenated_input_ids.device

    # Pre-compute valid sequence lengths and total size
    seq_lens = torch.tensor(sequence_lengths, dtype=torch.int32, device=device)
    mask = seq_lens > 1
    input_lengths_tensor = seq_lens[mask] - 1
    total_length = input_lengths_tensor.sum().item()

    model_inputs = torch.empty(total_length, dtype=concatenated_input_ids.dtype, device=device)
    model_targets = torch.empty(total_length, dtype=concatenated_input_ids.dtype, device=device)

    offset = 0
    batch_size = len(cu_seqlens) - 1
    for i in range(batch_size):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_length = end_idx - start_idx

        if seq_length > 1:
            length = seq_length - 1
            model_inputs[offset:offset + length] = concatenated_input_ids[start_idx:end_idx-1]
            model_targets[offset:offset + length] = concatenated_input_ids[start_idx+1:end_idx]
            offset += length

    cu_seqlens_shifted = torch.zeros(len(input_lengths_tensor) + 1, dtype=torch.int32, device=device)
    torch.cumsum(input_lengths_tensor, dim=0, out=cu_seqlens_shifted[1:])
    max_seqlen_shifted = input_lengths_tensor.max().item()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        embeddings = model.get_embeddings(model_inputs, cu_seqlens=cu_seqlens_shifted, max_seqlen=max_seqlen_shifted)

    classifier_weights = model.get_classifier_weights()

    # linear_cross_entropy(..., shift=1) incompatible with varlen because it would create
    # invalid cross-sequence predictions. E.g: batch ["cat sat", "dog ran"]
    # concatenated as [cat, sat, dog, ran]. shift=1 would predict dog from sat
    # across the sequence boundary. Manual shifting ensures each sequence only predicts
    # its own continuation: [cat]->[sat], [dog]->[ran], never [sat]->[dog].
    lm_loss, lse = linear_cross_entropy(embeddings, classifier_weights, model_targets, return_lse=True)
    z_loss = compute_z_loss(lse, model_targets, shift=0)

    return lm_loss + z_loss * 1e-7, lm_loss

def build_muon_optimizer(model, muon_lr=0.02, adam_lr=2e-4):
    muon_params = []

    zero_centered_rmsnorm_params = []
    tversky_scalars = []  # alpha, beta, theta
    adam_decay_params = []
    adam_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = model.get_submodule('.'.join(name.split('.')[:-1]))
        param_name = name.split('.')[-1]

        # Embedding and output layer -> Adam (no decay)
        if 'embedding' in name or 'output_layer' in name:
            adam_no_decay_params.append(param)

        # ZeroCenteredRMSNorm weight (1D) -> Adam
        elif isinstance(module, ZeroCenteredRMSNorm):
            zero_centered_rmsnorm_params.append(param)

        elif isinstance(module, TverskyLayer):
            if param_name in ['alpha', 'beta', 'theta']:
                tversky_scalars.append(param)
            elif param_name in ['prototypes', 'features']:
                # 2D matrices -> Muon
                muon_params.append(param)
            else:
                adam_decay_params.append(param)

        # Shared tversky features (2D) -> Muon
        elif 'shared_features' in name:
            muon_params.append(param)

        # All Linear layer weights (2D, bias=False) -> Muon
        elif isinstance(module, nn.Linear) and param_name == 'weight':
            muon_params.append(param)

        elif param.ndim < 2:
            adam_no_decay_params.append(param)

        elif param.ndim >= 2:
            muon_params.append(param)

        else:
            adam_decay_params.append(param)

    param_groups = [
        # Muon for all 2D hidden weights
        dict(params=muon_params, use_muon=True,
             lr=muon_lr, momentum=0.95, beta2=0.95, weight_decay=1e-4),
        # dict(params=muon_params, use_muon=True,
        #      lr=muon_lr, momentum=0.95, weight_decay=1e-4),

        # Adam groups for all the other things
        dict(params=zero_centered_rmsnorm_params, use_muon=False,
             lr=adam_lr, betas=(0.9, 0.95), eps=1e-16, weight_decay=1e-4),
        # Slow down tversky scalar overcompensation as this tends to be the overcorrecting term
        dict(params=tversky_scalars, use_muon=False,
             lr=2e-7, betas=(0.9, 0.95), eps=1e-16),
        dict(params=adam_decay_params, use_muon=False,
             lr=adam_lr, betas=(0.9, 0.95), eps=1e-16, weight_decay=1e-5),
        dict(params=adam_no_decay_params, use_muon=False,
             lr=adam_lr, betas=(0.9, 0.95), eps=1e-16, weight_decay=0.0),
    ]

    print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print(f"Adam RMSNorm: {sum(p.numel() for p in zero_centered_rmsnorm_params):,}")
    print(f"Adam Tversky scalars: {sum(p.numel() for p in tversky_scalars):,}")
    print(f"Adam decay: {sum(p.numel() for p in adam_decay_params):,}")
    print(f"Adam no decay: {sum(p.numel() for p in adam_no_decay_params):,}")

    return SingleDeviceNorMuonWithAuxAdam(param_groups)

def build_weight_decay_optm(model, muon_lr, adam_lr):
    zero_centered_rmsnorm_params = []
    decay_params = []
    no_decay_params = []

    # We want to be very careful about what is decayed
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = model.get_submodule('.'.join(name.split('.')[:-1]))

        if isinstance(module, ZeroCenteredRMSNorm):
            # Explicitly decay Zero centered RMS
            zero_centered_rmsnorm_params.append(param)
        elif any(exclude in name for exclude in [
            'bias', 'embedding', 'output_layer', 'norm.weight',
            'layernorm', 'dt_bias', 'A_log', 'expert_biases'
        ]):
            # Exclude a variety of common layers such as embedding and output we want to avoid normalizing.
            no_decay_params.append(param)
        else:
            # Everything else.
            decay_params.append(param)

    return torch.optim.Adam([
        {'params': zero_centered_rmsnorm_params, 'weight_decay': 1e-4},
        {'params': decay_params, 'weight_decay': 1e-5},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=adam_lr, eps=1e-16) # EPS recommended by deep seek v3 paper I think?

def defrag_cuda():
    import gc
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def train(model, train_dataset, tokenizer, num_epochs=1, batch_size=72, learning_rate=1e-4):
    device = torch.device("cuda")
    model.to(device)
    optimizer = build_muon_optimizer(model, muon_lr=1e-3, adam_lr=learning_rate)

    logger = TBLogger(
        log_dir='logs/current_run',
        flush_secs=10,
        enable_detailed_logging=True,
        detailed_frequency=10
    )

    count_parameters_layerwise(model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=varlen_collate_fn,
        pin_memory=True,
    )

    global_step = 0
    total_tokens = 0
    epoch_tokens = 0

    defrag_cuda()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_tokens += epoch_tokens
        epoch_tokens = 0

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            cu_seqlens = batch["cu_seqlens"].to(device)
            max_seqlen = batch["max_seqlen"]
            sequence_lengths = batch["sequence_lengths"]

            optimizer.zero_grad()

            loss, language_loss = compute_varlen_loss_with_cce(model, input_ids, cu_seqlens, sequence_lengths)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_tokens = input_ids.size(0)
            epoch_tokens += num_tokens

            logger.log_training_metrics(
                loss=loss.detach(),
                optimizer=optimizer,
                global_step=global_step,
                epoch=epoch,
                batch_idx=batch_idx,
                num_tokens=num_tokens
            )

            logger.log({}, step=global_step, model=model, detailed_logging=False)

            global_step += 1

            if global_step % 20 == 0:
                avg_seq_len = sum(sequence_lengths) / len(sequence_lengths)
                ppl = math.exp(language_loss.item())
                print(f"Step {global_step}, Loss +Z losses: {loss.item():.4f}, Language Loss: {language_loss.item():.4f}, PPL: {ppl:.2f}, Avg Seq Len: {avg_seq_len:.1f}, Tokens: {num_tokens}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Epoch Total Tokens: {epoch_tokens}")

        logger.log({
            'epoch/avg_loss': avg_loss,
            'epoch/total_tokens': epoch_tokens
        }, step=global_step)


def main():
    config = ModelConfig()

    train_dataset = JSONLDirectoryTokenizedDataset('dataset')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = CoolLanguageModelWowExclamationMark(config)

    torch.compile(model, mode="max-autotune")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_dataset, tokenizer)
    save_model(model, "model.safetensors")

if __name__ == "__main__":
    main()
