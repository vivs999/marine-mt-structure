"""Curated UniProt taxonomy allowlist for marine metallothionein ingestion.

The PRD calls for marine fish and invertebrate groups with documented MT
sequences. This helper keeps the allowlist in one place so the UniProt query
builder and the CLI default stay aligned.
"""

from __future__ import annotations

MARINE_TAXON_GROUPS: tuple[tuple[str, int], ...] = (
    ("Actinopterygii", 7898),
    ("Mollusca", 6447),
    ("Crustacea", 6657),
    ("Cnidaria", 6073),
    ("Echinodermata", 7586),
)


def get_marine_taxon_ids() -> list[int]:
    return [taxon_id for _, taxon_id in MARINE_TAXON_GROUPS]
