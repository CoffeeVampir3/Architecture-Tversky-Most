import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

class JSONLDirectoryTokenizedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.sequences = []

        brick_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if not brick_dirs:
            raise ValueError(f"No brick directories found in {data_dir}")

        jsonl_files = []
        for brick_dir in brick_dirs:
            jsonl_files.extend(sorted(brick_dir.glob('*.jsonl')))

        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in brick directories under {data_dir}")

        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line)
                    token_ids = chunk['token_ids']
                    if len(token_ids) <= 10: continue
                    self.sequences.append(torch.tensor(token_ids, dtype=torch.int32))

        if not self.sequences:
            raise ValueError(f"No sequences loaded from {data_dir}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def varlen_collate_fn(batch):
    sequence_lengths = [len(seq) for seq in batch]
    concatenated = torch.cat(batch, dim=0)
    cu_seqlens = torch.zeros(len(batch) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(torch.tensor(sequence_lengths, dtype=torch.int32), dim=0)
    return {
        'input_ids': concatenated,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max(sequence_lengths),
        'sequence_lengths': sequence_lengths,
        'batch_size': len(batch)
    }
