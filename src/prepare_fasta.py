"""Convert UniProt sequence rows into FASTA format.

This utility is intentionally small and dependency-light so it can be used both
as a CLI step and as a library helper for tests and notebooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

from src.ingest_uniprot import read_sequences_csv


def sequence_rows_to_fasta(rows: Iterable[Dict[str, str]]) -> str:
    lines: List[str] = []
    for row in rows:
        accession = str(row.get("uniprot_id", "unknown"))
        organism = str(row.get("organism", "unknown organism"))
        sequence = str(row.get("sequence", ""))
        if not sequence:
            continue
        header = f">{accession} {organism}".strip()
        lines.append(header)
        lines.append(sequence)
    return "\n".join(lines) + ("\n" if lines else "")


def write_fasta(rows: Iterable[Dict[str, str]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(sequence_rows_to_fasta(rows))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write UniProt sequence rows to FASTA.")
    parser.add_argument("--input", default="data/raw/uniprot_sequences.csv")
    parser.add_argument("--output", default="data/raw/sequences.fasta")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = read_sequences_csv(args.input)
    write_fasta(rows, args.output)
    print(f"Wrote FASTA for {len(rows)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
