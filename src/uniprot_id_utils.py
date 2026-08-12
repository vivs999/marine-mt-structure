"""Utilities for extracting canonical UniProt accessions from noisy IDs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Optional

# UniProtKB accession formats:
# - 6 chars: [OPQ][0-9][A-Z0-9]{3}[0-9]
# - 6/10 chars: [A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}
UNIPROT_ACCESSION_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"("
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
    r")"
    r"(?![A-Z0-9])"
)


def _candidate_strings(raw_value: str) -> Iterable[str]:
    """Yield normalized candidate strings likely to contain a UniProt ID."""
    text = str(raw_value).strip()
    if not text:
        return []

    path_name = Path(text).name
    stem = Path(path_name).stem
    stem_without_suffixes = re.sub(r"(\.pae|_scores|_r\d+_len\d+)$", "", stem, flags=re.IGNORECASE)

    candidates = [
        text,
        text.upper(),
        path_name,
        path_name.upper(),
        stem,
        stem.upper(),
        stem_without_suffixes,
        stem_without_suffixes.upper(),
    ]

    split_tokens = [t for t in re.split(r"[^A-Za-z0-9]+", text) if t]
    candidates.extend(split_tokens)
    candidates.extend(token.upper() for token in split_tokens)
    return candidates


def extract_canonical_uniprot_id(raw_value: str) -> Optional[str]:
    """Extract canonical UniProt accession from a noisy identifier or path.

    Examples:
        - ``Q7ZSY6_r3_len60`` -> ``Q7ZSY6``
        - ``.../P68503_95ef7/P68503_r3_len61.pdb`` -> ``P68503``
        - ``P52722`` -> ``P52722``
    """
    for candidate in _candidate_strings(raw_value):
        match = UNIPROT_ACCESSION_PATTERN.search(candidate)
        if match:
            return match.group(1).upper()
    return None


def normalize_summary_csv_ids(summary_csv_path: Path) -> int:
    """Normalize ``id`` values in an ESMFold summary CSV in place.

    The function preserves the existing columns/order and only rewrites rows where
    a canonical UniProt accession can be extracted.

    Returns:
        Number of rows that were updated.
    """
    with summary_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "id" not in fieldnames:
        return 0

    updated_count = 0
    for row in rows:
        row_id = (row.get("id") or "").strip()
        canonical = extract_canonical_uniprot_id(row_id)
        if canonical is None:
            for key in ("pdb", "pae", "scores"):
                path_value = row.get(key)
                if not path_value:
                    continue
                canonical = extract_canonical_uniprot_id(path_value)
                if canonical is not None:
                    break
        if canonical and canonical != row_id:
            row["id"] = canonical
            updated_count += 1

    temp_path = summary_csv_path.with_suffix(summary_csv_path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(summary_csv_path)
    return updated_count
