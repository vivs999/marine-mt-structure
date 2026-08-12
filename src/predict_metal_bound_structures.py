#!/usr/bin/env python3
"""
Predict Zn²⁺-bound MT structures using Boltz-1 (CPU/MPS capable).

Metallothioneins are IDPs that only adopt a defined fold upon metal coordination.
ESMFold apo predictions are meaningless for MTs (pLDDT ~44, random coil).
This script generates biologically accurate metal-bound structures by explicitly
supplying Zn²⁺ ions matched to each sequence's cysteine count.

Metal stoichiometry rule (from crystal structure literature):
  α-cluster: Zn₄Cys₁₁  ->  2.75 Cys per Zn
  β-cluster: Zn₃Cys₉   ->  3.00 Cys per Zn
  Overall average: ~3.0 Cys per Zn²⁺ in bridged tetrahedral clusters
  n_zn = max(1, round(n_cys / 3.0)), capped at 12

Usage:
  uv run --python 3.12 --with boltz python3 src/predict_metal_bound_structures.py

Runtime estimate on Apple Silicon CPU (~100 aa, 25 sampling steps):
  ~2-5 min/sequence  x  535 sequences  ≈  18-45 h total
  Tip: run overnight -> logs/boltz_metalbound.log

Output:
  data/raw/pdb_structures_metalbound/{uniprot_id}/{uniprot_id}_model_0.cif
"""

from __future__ import annotations
import csv
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml  # pip install pyyaml (or uv run --with pyyaml)

FASTA_PATH  = Path("data/processed/sequences.fasta")
OUT_STRUCT  = Path("data/raw/pdb_structures_metalbound")
BOLTZ_CACHE = Path("~/.boltz").expanduser()
LOG_CSV     = Path("logs/boltz_progress.csv")

# Boltz prediction settings - optimised for CPU speed on small proteins.
# Raise sampling_steps to 100-200 for publication-quality structures at the
# cost of ~4-8x longer runtime.
BOLTZ_SAMPLING_STEPS = 25
BOLTZ_RECYCLING_STEPS = 3
BOLTZ_DIFFUSION_SAMPLES = 1
BOLTZ_ACCELERATOR = "gpu"      # uses MPS on Apple Silicon; change to "cpu" if issues
BOLTZ_MODEL = "boltz1"         # boltz1 is faster than boltz2
WORKERS = 3                    # MPS is a single device; 3 workers saturates it without contention

# Zn²⁺ SMILES string (tetrahedral coordination, 2+ charge)
ZN_SMILES = "[Zn+2]"

# ASCII letter sequence for chain IDs (protein=A, metals=B,C,D,...)
_CHAIN_LETTERS = "BCDEFGHIJKLMNOPQRSTUVWXYZ"


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


def n_zinc_ions(seq: str) -> int:
    """
    Estimate Zn²⁺ stoichiometry from cysteine count.

    Bridged tetrahedral clusters in class I MTs give ~3.0 Cys per Zn:
      mammalian MT-2  : 20 Cys -> 7 Zn  (ratio 2.86)
      sea urchin MT   : 18 Cys -> 6 Zn  (ratio 3.00)
      oyster MT       : 20 Cys -> 6 Zn  (ratio 3.33) - slightly Cd-preferring

    Lower Cys sequences (< 6) get 1 Zn minimum.
    """
    n_cys = seq.count("C")
    n_zn = max(1, round(n_cys / 3.0))
    return min(n_zn, 12)   # cap at reasonable maximum


def make_boltz_yaml(uid: str, seq: str, n_zn: int) -> dict:
    """Build a Boltz YAML dict for protein + n_zn Zn²⁺ ions."""
    sequences = [{"protein": {"id": "A", "sequence": seq}}]
    for i in range(n_zn):
        chain_id = _CHAIN_LETTERS[i]
        sequences.append({"ligand": {"id": chain_id, "smiles": ZN_SMILES}})
    return {"version": 1, "sequences": sequences}


def run_boltz(uid: str, yaml_path: Path, out_dir: Path) -> bool:
    """Run boltz predict for one sequence. Returns True on success."""
    # boltz installs a CLI entry point in the same bin dir as the interpreter
    boltz_exe = str(Path(sys.executable).parent / "boltz")
    cmd = [
        boltz_exe, "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--accelerator", BOLTZ_ACCELERATOR,
        "--model", BOLTZ_MODEL,
        "--sampling_steps", str(BOLTZ_SAMPLING_STEPS),
        "--recycling_steps", str(BOLTZ_RECYCLING_STEPS),
        "--diffusion_samples", str(BOLTZ_DIFFUSION_SAMPLES),
        "--output_format", "mmcif",
        "--use_msa_server",
        "--override",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    BOLTZ ERROR for {uid}:\n{result.stderr[-500:]}")
        return False
    return True


def already_done(uid: str) -> bool:
    """Check if Boltz CIF already exists. Path: boltz_results_{uid}/predictions/{uid}/{uid}_model_0.cif"""
    cif = OUT_STRUCT / uid / f"boltz_results_{uid}" / "predictions" / uid / f"{uid}_model_0.cif"
    return cif.exists()


def process_one(uid: str, seq: str, total: int, counter: list, lock: threading.Lock) -> dict:
    """Prepare input, run Boltz, return log row. Runs in a thread worker."""
    n_cys = seq.count("C")
    n_zn  = n_zinc_ions(seq)

    uid_dir = OUT_STRUCT / uid
    uid_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = uid_dir / f"{uid}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(make_boltz_yaml(uid, seq, n_zn), f, default_flow_style=False)

    t0 = time.time()
    ok = run_boltz(uid, yaml_path, uid_dir)
    elapsed = round(time.time() - t0, 1)

    with lock:
        counter[0] += 1
        done = counter[0]
        status = "ok" if ok else "failed"
        print(f"[{done}/{total}] {uid}  Cys={n_cys}  Zn²⁺={n_zn}  {status}  ({elapsed}s)",
              flush=True)

    return {"uniprot_id": uid, "n_cys": n_cys, "n_zn": n_zn,
            "status": "ok" if ok else "failed", "elapsed_s": elapsed}


def main():
    import importlib.util
    if importlib.util.find_spec("yaml") is None:
        print("ERROR: pyyaml not found. Run with: uv run --python 3.12 --with boltz --with pyyaml ...")
        sys.exit(1)

    sequences = parse_fasta(FASTA_PATH)
    print(f"Sequences loaded: {len(sequences)}")

    OUT_STRUCT.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    LOG_CSV.parent.mkdir(exist_ok=True)

    # Skip already-completed structures (resume support)
    todo = [(uid, seq) for uid, seq in sequences if not already_done(uid)]
    skipped = len(sequences) - len(todo)
    if skipped:
        print(f"Resuming: {skipped} already done, {len(todo)} remaining")
    print(f"Running {WORKERS} parallel workers\n")

    log_fields = ["uniprot_id", "n_cys", "n_zn", "status", "elapsed_s"]
    log_mode = "a" if LOG_CSV.exists() else "w"
    log_fh   = open(LOG_CSV, log_mode, newline="")
    log_writer = csv.DictWriter(log_fh, fieldnames=log_fields)
    if log_mode == "w":
        log_writer.writeheader()

    lock: threading.Lock = threading.Lock()
    counter: list[int]   = [0]          # mutable int shared across threads
    success = failed = 0
    total = len(todo)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(process_one, uid, seq, total, counter, lock): uid
            for uid, seq in todo
        }
        for fut in as_completed(futures):
            try:
                row = fut.result()
                with lock:
                    log_writer.writerow(row)
                    log_fh.flush()
                if row["status"] == "ok":
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                uid = futures[fut]
                print(f"  EXCEPTION {uid}: {e}")
                failed += 1

    log_fh.close()
    print(f"\nDone: {success} new  |  {skipped} skipped  |  {failed} failed")
    print(f"Structures -> {OUT_STRUCT}/")
    print(f"Progress log -> {LOG_CSV}")


if __name__ == "__main__":
    main()
