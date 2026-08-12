"""
Phase 2 - Batch structural feature extraction for all 532 ESMFold PDB files.

Recursively searches --pdb-dir for *.pdb files organized in the ESMFold
prediction directory layout:
  {batch}/predictions/{ID}_{hash}/{ID}_r12_len{N}.pdb

Parses the UniProt ID from each filename (everything before _r12_), extracts
structural features, joins metadata from --metadata CSV, and writes the
integrated dataset to --output.

Usage:
  uv run python3 src/batch_extract_features.py \\
    --pdb-dir data/raw/pdb_structures \\
    --metadata data/processed/metadata_enriched_v2.csv \\
    --output data/processed/integrated_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.feature_extraction import extract_features_from_pdb_and_sequence, parse_sequence_from_pdb


def parse_uniprot_id(stem: str) -> str:
    """Extract UniProt accession from ESMFold filename stem.
    E.g. 'A0A6G5YPG2_r12_len74' -> 'A0A6G5YPG2'
    """
    m = re.match(r"^([A-Za-z0-9]+)_r\d+_len\d+$", stem)
    if m:
        return m.group(1)
    # Fallback: everything before first underscore followed by 'r' digit
    parts = stem.split("_r")
    return parts[0]


def find_pdb_files(pdb_dir: Path) -> list[tuple[str, Path]]:
    """Return (uniprot_id, pdb_path) for all *.pdb files under pdb_dir."""
    found = []
    for pdb_path in sorted(pdb_dir.rglob("*.pdb")):
        uid = parse_uniprot_id(pdb_path.stem)
        found.append((uid, pdb_path))
    return found


def load_metadata(metadata_csv: Path) -> dict[str, dict]:
    """Load metadata keyed by uniprot_id. Returns {} if file doesn't exist."""
    if not metadata_csv.exists():
        print(f"  WARNING: metadata file not found: {metadata_csv}")
        return {}
    meta: dict[str, dict] = {}
    with open(metadata_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            uid = row.get("uniprot_id", "")
            if uid:
                meta[uid] = dict(row)
    return meta


def extract_all(
    pdb_dir: Path,
    metadata_csv: Optional[Path],
    output_csv: Path,
) -> int:
    print(f"Scanning PDB directory: {pdb_dir}")
    entries = find_pdb_files(pdb_dir)
    print(f"  Found {len(entries)} PDB files")

    meta = load_metadata(metadata_csv) if metadata_csv else {}
    if meta:
        print(f"  Metadata loaded: {len(meta)} entries")

    rows: list[dict] = []
    n_meta_matched = 0

    for i, (uid, pdb_path) in enumerate(entries):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(entries)}] processing...")

        try:
            pdb_text = pdb_path.read_text()
            sequence = parse_sequence_from_pdb(pdb_text)
            feats = extract_features_from_pdb_and_sequence(pdb_text, sequence=sequence or None)
        except Exception as e:
            print(f"  WARNING: failed to extract features from {pdb_path.name}: {e}")
            feats = {}

        row: dict = {"uniprot_id": uid}
        row.update({k: ("" if v is None else str(v)) for k, v in feats.items()})

        if uid in meta:
            n_meta_matched += 1
            for k, v in meta[uid].items():
                if k != "uniprot_id":
                    row.setdefault(k, v)

        rows.append(row)

    if meta:
        print(f"  Metadata matched: {n_meta_matched}/{len(entries)} sequences")

    # Collect all field names; uniprot_id first, then structural, then metadata
    structural_cols = sorted({
        k for r in rows for k in r
        if k not in {"uniprot_id"} and k not in (meta or {}).get(
            next(iter(meta)) if meta else "", {}
        )
    })
    meta_cols: list[str] = []
    if meta:
        sample_meta = next(iter(meta.values()))
        meta_cols = [k for k in sample_meta if k != "uniprot_id"]

    all_fields_set = {"uniprot_id"} | set(structural_cols) | set(meta_cols)
    other_cols = sorted(k for r in rows for k in r if k not in all_fields_set)
    fieldnames = ["uniprot_id"] + structural_cols + meta_cols + other_cols

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for f in fieldnames:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    fieldnames = deduped

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({f: row.get(f, "") for f in fieldnames})

    print(f"\nFeatures extracted: {len(rows)} sequences")
    print(f"Columns: {len(fieldnames)}")
    print(f"Output: {output_csv}")
    return len(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract structural features from ESMFold PDB files.")
    p.add_argument("--pdb-dir",  default="data/raw/pdb_structures",
                   help="Root directory containing ESMFold prediction subdirectories")
    p.add_argument("--metadata", default=None,
                   help="Path to enriched metadata CSV to join on uniprot_id")
    p.add_argument("--output",   default="data/processed/integrated_v2.csv",
                   help="Output CSV path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    n = extract_all(
        pdb_dir=Path(args.pdb_dir),
        metadata_csv=Path(args.metadata) if args.metadata else None,
        output_csv=Path(args.output),
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
