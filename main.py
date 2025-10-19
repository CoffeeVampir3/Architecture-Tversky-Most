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
from muon import SingleDeviceNorMuonWithAuxAdam

from utils.jsonl_dataloader import JSONLDirectoryTokenizedDataset, varlen_collate_fn
from utils.trainutils import count_parameters_layerwise, TBLogger, save_checkpoint, load_checkpoint
from modeling.model import CoolLanguageModelWowExclamationMark, ModelConfig
from modeling.zRMSNorm import ZeroCenteredRMSNorm

import os
import torch._inductor.config

# There's a memory leak related to very specific interactions of AMP + flash attn varlen + Tversky O projections.
# The exact nature is unclear to me but these settings (specifically the dyanic graph one) prevents the leak.
# The expandable segments seems to reduce the overall allocation size
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
#torch._inductor.config.triton.cudagraph_trees = False

#torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.fp32_precision = 'ieee'

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
    adam_1d_params = []
    adam_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Embedding and output layer -> Adam (no decay)
        if 'embedding' in name or 'output_layer' in name:
            adam_no_decay_params.append(param)
        # All 2D parameters -> Muon
        elif param.ndim >= 2:
            muon_params.append(param)
        # All 1D parameters -> Adam with decay
        elif param.ndim == 1:
            adam_1d_params.append(param)
        else:
            # Scalars (0D) if any -> Adam
            adam_1d_params.append(param)

    param_groups = [
        dict(params=muon_params, use_muon=True,
             lr=muon_lr, momentum=0.95, beta2=0.95, weight_decay=1e-1),
        dict(params=adam_1d_params, use_muon=False,
             lr=adam_lr, betas=(0.2, 0.95), eps=1e-16, weight_decay=1e-5),
        dict(params=adam_no_decay_params, use_muon=False,
             lr=adam_lr, betas=(0.2, 0.95), eps=1e-16, weight_decay=0.0),
    ]

    print(f"Muon params (2D): {sum(p.numel() for p in muon_params):,}")
    print(f"Adam 1D params: {sum(p.numel() for p in adam_1d_params):,}")
    print(f"Adam no decay (embeddings): {sum(p.numel() for p in adam_no_decay_params):,}")

    return SingleDeviceNorMuonWithAuxAdam(param_groups)

def defrag_cuda():
    import gc
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def save_rolling_checkpoint(model, optimizer, global_step, epoch, checkpoint_dir="checkpoints", keep_n=5):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}")
    save_checkpoint(model, optimizer, global_step, epoch, checkpoint_path)

    all_files = os.listdir(checkpoint_dir)
    checkpoint_steps = set()
    for f in all_files:
        if f.startswith("step_") and ("_model.safetensors" in f or "_optimizer.pt" in f):
            step_num = int(f.split('_')[1])
            checkpoint_steps.add(step_num)

    sorted_steps = sorted(checkpoint_steps)
    if len(sorted_steps) > keep_n:
        for old_step in sorted_steps[:-keep_n]:
            for ext in ["_model.safetensors", "_optimizer.pt"]:
                file_to_remove = os.path.join(checkpoint_dir, f"step_{old_step}{ext}")
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)

def train(model, train_dataset, tokenizer, num_epochs=1, batch_size=24, learning_rate=1e-4, accumulation_steps=3):
    device = torch.device("cuda")
    model.to(device)
    optimizer = build_muon_optimizer(model, muon_lr=1e-3, adam_lr=4e-4)
    optimizer.preallocate_state()
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
    )
    global_step = 0
    total_tokens = 0
    epoch_tokens = 0
    defrag_cuda()
    os.makedirs("checkpoints", exist_ok=True)

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
            num_tokens = input_ids.size(0)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, language_loss = compute_varlen_loss_with_cce(model, input_ids, cu_seqlens, sequence_lengths)
                loss = loss / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                total_loss += (loss.item() * accumulation_steps)

                logger.log_training_metrics(
                    loss=loss * accumulation_steps,
                    optimizer=optimizer,
                    global_step=global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    num_tokens=num_tokens
                )
                logger.log({}, step=global_step, model=model, detailed_logging=False)

                if global_step % 500 == 0:
                    save_rolling_checkpoint(model, optimizer, global_step, epoch, checkpoint_dir="checkpoints", keep_n=5)

                if global_step % 20 == 0:
                    avg_seq_len = sum(sequence_lengths) / len(sequence_lengths)
                    ppl = math.exp(language_loss.item())
                    print(f"Step {global_step}, Loss +Z losses: {(loss.item() * accumulation_steps):.4f}, Language Loss: {language_loss.item():.4f}, PPL: {ppl:.2f}, Avg Seq Len: {avg_seq_len:.1f}, Tokens: {num_tokens}")

                global_step += 1

            loss = loss.detach()
            language_loss = language_loss.detach()
            epoch_tokens += num_tokens

        if (len(train_loader) % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / (len(train_loader) // accumulation_steps)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Epoch Total Tokens: {epoch_tokens}")
        logger.log({
            'epoch/avg_loss': avg_loss,
            'epoch/total_tokens': epoch_tokens
        }, step=global_step)

    save_checkpoint(model, optimizer, global_step, epoch, "checkpoints/final")

def main():
    config = ModelConfig()

    train_dataset = JSONLDirectoryTokenizedDataset('dataset')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = CoolLanguageModelWowExclamationMark(config)

    model.compile(mode="max-autotune")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_dataset, tokenizer)

if __name__ == "__main__":
    main()
