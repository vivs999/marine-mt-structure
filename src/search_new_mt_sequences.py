#!/usr/bin/env python3
"""
Search UniProt for marine metallothionein sequences not yet in our dataset.
Covers fish, invertebrates, marine mammals, algae, hydrothermal vent organisms,
and MT subtypes (MT-A, MT-B, MT-C, MT-like, phytochelatin synthase excluded).

Output:
  data/raw/new_mt_candidates_expanded.csv  -- all new candidates with organism/habitat notes
"""

from __future__ import annotations
import csv, time, requests
from pathlib import Path

EXISTING_CSV = Path("data/processed/metadata_populated.csv")
OUT_CSV = Path("data/raw/new_mt_candidates_expanded.csv")
API = "https://rest.uniprot.org/uniprotkb/search"
HEADERS = {"User-Agent": "mtsef-research/1.0"}

# taxonomy groups to query
# taxon ID, common label, expected habitat
TAXA = [
    # Fish - marine teleosts
    ("7897",  "Latimeria (coelacanth)",        "open_ocean"),
    ("7898",  "Actinopterygii (ray-fin fish)",  "coastal_estuarine"),
    ("8492",  "Elasmobranchii (sharks/rays)",   "open_ocean"),
    # Marine invertebrates
    ("6657",  "Crustacea",                      "coastal_estuarine"),
    ("6447",  "Mollusca",                       "coastal_estuarine"),
    ("7586",  "Echinodermata",                  "open_ocean"),
    ("6340",  "Annelida (polychaetes etc.)",    "coastal_estuarine"),
    ("6073",  "Cnidaria (coral/jellyfish)",     "coastal_estuarine"),
    ("6040",  "Porifera (sponges)",             "coastal_estuarine"),
    ("10197", "Tunicata (sea squirts)",         "coastal_estuarine"),
    # Marine mammals
    ("9721",  "Cetacea (whales/dolphins)",      "open_ocean"),
    ("9700",  "Pinnipedia (seals/sea lions)",   "open_ocean"),
    ("9813",  "Sirenia (manatees/dugongs)",     "coastal_estuarine"),
    # Marine/coastal birds
    ("8782",  "Aves (seabirds)",                "open_ocean"),
    # Marine algae / plants
    ("2836",  "Rhodophyta (red algae)",         "coastal_estuarine"),
    ("3041",  "Chlorophyta (green algae)",      "coastal_estuarine"),
    ("35493", "Stramenopiles (brown algae etc)","coastal_estuarine"),
    ("4232",  "Zostera/seagrass",               "coastal_estuarine"),
    # Hydrothermal vent / deep sea
    ("6154",  "Platyhelminthes",               "open_ocean"),
    ("6231",  "Nematoda (marine)",              "open_ocean"),
]

# MT keyword variants
MT_TERMS = [
    "metallothionein",
    "metallothionein-A",
    "metallothionein-B",
    "metallothionein-C",
    "metallothionein-like",
    "MT-A", "MT-B", "MT-C",
]


def load_existing_ids() -> set[str]:
    import csv as _csv
    ids = set()
    with open(EXISTING_CSV) as fh:
        for row in _csv.DictReader(fh):
            ids.add(row["uniprot_id"])
    return ids


def search_uniprot(taxon_id: str, term: str, size: int = 500) -> list[dict]:
    query = f'protein_name:"{term}" AND taxonomy_id:{taxon_id}'
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,organism_name,protein_name,length,sequence,reviewed",
        "size": size,
    }
    try:
        r = requests.get(API, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []
        lines = r.text.strip().splitlines()
        if len(lines) < 2:
            return []
        hdrs = lines[0].split("\t")
        rows = []
        for line in lines[1:]:
            vals = line.split("\t")
            if len(vals) == len(hdrs):
                rows.append(dict(zip(hdrs, vals)))
        return rows
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def is_valid_mt(row: dict) -> bool:
    """Basic filter: must have sequence, reasonable MT length (40-120 aa), contain cysteines."""
    seq = row.get("Sequence", "")
    if not seq or len(seq) < 40 or len(seq) > 120:
        return False
    cys = seq.count("C")
    if cys < 6:  # MTs have ≥6-7 Cys
        return False
    # Exclude phytochelatin synthase, glutathione, etc.
    name = row.get("Protein names", "").lower()
    if any(x in name for x in ["synthase", "glutathione", "peroxidase", "reductase"]):
        return False
    return True


def main():
    existing_ids = load_existing_ids()
    print(f"Existing dataset: {len(existing_ids)} sequences")
    print(f"Searching {len(TAXA)} taxonomic groups x {len(MT_TERMS)} MT terms...\n")

    found: dict[str, dict] = {}  # accession -> row

    for taxon_id, taxon_label, habitat_hint in TAXA:
        group_new = 0
        for term in MT_TERMS:
            results = search_uniprot(taxon_id, term)
            for row in results:
                acc = row.get("Entry", "").strip()
                if not acc or acc in existing_ids or acc in found:
                    continue
                if not is_valid_mt(row):
                    continue
                found[acc] = {
                    "uniprot_id": acc,
                    "organism": row.get("Organism", ""),
                    "protein_name": row.get("Protein names", ""),
                    "length": row.get("Length", ""),
                    "sequence": row.get("Sequence", ""),
                    "reviewed": row.get("Reviewed", ""),
                    "taxon_group": taxon_label,
                    "habitat_hint": habitat_hint,
                    "search_term": term,
                }
                group_new += 1
            time.sleep(0.2)

        if group_new:
            print(f"  {taxon_label}: +{group_new} new sequences")

    print(f"\nTotal new candidates: {len(found)}")

    if not found:
        print("No new sequences found.")
        return

    # Write output
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["uniprot_id", "organism", "protein_name", "length", "sequence",
              "reviewed", "taxon_group", "habitat_hint", "search_term"]
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(found.values())

    print(f"Written -> {OUT_CSV}")

    # Summary by group
    import collections
    by_group = collections.Counter(r["taxon_group"] for r in found.values())
    print("\nBreakdown by taxonomic group:")
    for grp, cnt in by_group.most_common():
        print(f"  {cnt:4d}  {grp}")

    # Reviewed vs unreviewed
    rev = sum(1 for r in found.values() if r["reviewed"] == "reviewed")
    print(f"\nReviewed (Swiss-Prot): {rev}")
    print(f"Unreviewed (TrEMBL):   {len(found)-rev}")


if __name__ == "__main__":
    main()
