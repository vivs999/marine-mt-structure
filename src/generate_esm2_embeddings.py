#!/usr/bin/env python3
"""
Generate ESM-2 mean-pooled embeddings for all MT sequences.

Runs locally - no Colab needed. Uses HuggingFace transformers with
float16 inference. Automatically selects MPS (Apple Silicon), CUDA, or CPU.

Model options (set MODEL_NAME below):
  facebook/esm2_t36_3B_UR50D   - 3B params, 2560-dim  (recommended, ~5.5 GB fp16)
  facebook/esm2_t33_650M_UR50D - 650M params, 1280-dim (~1.3 GB fp16)

Usage:
  uv run --with transformers --with torch python3 src/generate_esm2_embeddings.py

Output:
  data/processed/esm2_embeddings.csv
"""

from __future__ import annotations
import csv
from pathlib import Path

import torch

MODEL_NAME = "facebook/esm2_t36_3B_UR50D"   # change to 650M if RAM is tight
FASTA_PATH = Path("data/processed/sequences.fasta")
OUT_CSV    = Path("data/processed/esm2_embeddings.csv")
BATCH_SIZE = 8    # reduce to 4 if you hit OOM
MAX_LEN    = 1022  # ESM-2 hard limit; MTs are 40-130 aa so this never triggers


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")
    return dev


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    seqs: list[tuple[str, str]] = []
    uid, buf = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if uid:
                seqs.append((uid, "".join(buf)))
            uid, buf = line[1:].strip(), []
        else:
            buf.append(line)
    if uid:
        seqs.append((uid, "".join(buf)))
    return seqs


def main():
    from transformers import EsmModel, EsmTokenizer

    device = pick_device()

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.eval().to(device)
    emb_dim = model.config.hidden_size
    print(f"Embedding dim: {emb_dim}")

    sequences = parse_fasta(FASTA_PATH)
    print(f"Sequences: {len(sequences)} ({min(len(s) for _,s in sequences)}-"
          f"{max(len(s) for _,s in sequences)} aa)\n")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    col_names = [f"esm2_{i}" for i in range(emb_dim)]

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uniprot_id"] + col_names)

        for batch_start in range(0, len(sequences), BATCH_SIZE):
            batch = sequences[batch_start : batch_start + BATCH_SIZE]
            seqs_text = [seq[:MAX_LEN] for _, seq in batch]

            inputs = tokenizer(
                seqs_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN + 2,  # +2 for BOS/EOS tokens
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # last_hidden_state: [batch, seq_len, emb_dim]
            # attention_mask: 1 for real tokens, 0 for padding
            token_embs  = outputs.last_hidden_state  # includes BOS/EOS
            attn_mask   = inputs["attention_mask"].unsqueeze(-1).float()

            # Exclude BOS (pos 0) and EOS from mean pool by zeroing them via
            # the attention mask - tokenizer already marks pad as 0; BOS/EOS
            # are marked 1. We need residue-only mean, so strip position 0
            # and find where EOS sits per sequence.
            residue_embs = token_embs[:, 1:, :]          # drop BOS
            residue_mask = attn_mask[:, 1:, :]            # matching mask slice

            # Zero out EOS position for each sequence
            for i, seq in enumerate(seqs_text):
                eos_pos = len(seq)  # 0-indexed after BOS removal; EOS is at this pos
                if eos_pos < residue_mask.shape[1]:
                    residue_mask[i, eos_pos, :] = 0.0

            mean_emb = (residue_embs * residue_mask).sum(dim=1) / residue_mask.sum(dim=1).clamp(min=1e-9)
            mean_emb = mean_emb.cpu().float().numpy()

            for i, (uid, _) in enumerate(batch):
                writer.writerow([uid] + mean_emb[i].tolist())

            done = batch_start + len(batch)
            print(f"  [{done}/{len(sequences)}]", end="\r", flush=True)

    print(f"\nSaved -> {OUT_CSV}")
    print(f"Shape: {len(sequences)} rows x {emb_dim + 1} cols (id + embeddings)")


if __name__ == "__main__":
    main()
