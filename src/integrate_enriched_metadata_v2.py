#!/usr/bin/env python3
"""
Phase 1.5 - Merge v2 environmental labels into metadata.

Uses environmental_labels_v2.csv (observed OBIS depth + depth-mean salinity/pH)
on top of the existing biooracle_environmental.csv, with provenance hierarchy:
  observed > obis_occurrence_depth > biooracle_depthmean > biooracle_surface > habitat_default

Output: data/processed/metadata_enriched_v3.csv
"""

from __future__ import annotations
import csv
from pathlib import Path

METADATA_CSV = Path("data/processed/metadata_populated.csv")
BIO_V1_CSV   = Path("data/processed/biooracle_environmental.csv")
V2_CSV       = Path("data/processed/environmental_labels_v2.csv")
OUT_CSV      = Path("data/processed/metadata_enriched_v3.csv")

HABITAT_DEFAULTS = {
    "coastal_estuarine":  {"sst_mean_c": 22.0, "salinity_mean_psu": 32.0, "ph_mean": 8.1,  "depth_mean_m": 30.0,   "do_mean_mlL": 6.5},
    "open_ocean":         {"sst_mean_c": 15.0, "salinity_mean_psu": 35.0, "ph_mean": 8.1,  "depth_mean_m": 200.0,  "do_mean_mlL": 7.0},
    "polar":              {"sst_mean_c": -1.5, "salinity_mean_psu": 34.0, "ph_mean": 8.0,  "depth_mean_m": 100.0,  "do_mean_mlL": 8.5},
    "hydrothermal_vent":  {"sst_mean_c": 4.0,  "salinity_mean_psu": 35.0, "ph_mean": 6.5,  "depth_mean_m": 2500.0, "do_mean_mlL": 3.0},
}


def load_by_organism(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            org = row.get("organism", "")
            if org:
                out[org] = row
    return out


def pick(v1_row, v2_row, field_v1, field_v2, habitat, default_key):
    """Return (value, provenance) using priority hierarchy."""
    # v2 source (observed depth / depth-mean layers)
    if v2_row:
        v = v2_row.get(field_v2, "")
        src = v2_row.get(f"{field_v2}_src", "biooracle_depthmean")
        if v:
            return str(v), src

    # v1 Bio-ORACLE surface
    if v1_row:
        v = v1_row.get(field_v1, "")
        if v:
            return str(v), "biooracle_surface"

    # habitat default
    defs = HABITAT_DEFAULTS.get(habitat, HABITAT_DEFAULTS["coastal_estuarine"])
    default = defs.get(default_key)
    if default is not None:
        return str(default), f"habitat_default:{habitat}"

    return "", "missing"


def main():
    metadata = []
    with open(METADATA_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        base_fields = list(reader.fieldnames)
        metadata = list(reader)

    bio_v1 = load_by_organism(BIO_V1_CSV)
    env_v2 = load_by_organism(V2_CSV)

    print(f"Metadata rows:       {len(metadata)}")
    print(f"Bio-ORACLE v1:       {len(bio_v1)} species")
    print(f"Environmental v2:    {len(env_v2)} species")

    new_cols = [
        "sst_mean_c",        "sst_mean_c_status",        "sst_mean_c_provenance",
        "salinity_mean_psu", "salinity_mean_psu_status", "salinity_mean_psu_provenance",
        "ph_mean",           "ph_mean_status",           "ph_mean_provenance",
        "depth_mean_m",      "depth_mean_m_status",      "depth_mean_m_provenance",
        "do_mean_mlL",       "do_mean_mlL_status",       "do_mean_mlL_provenance",
    ]
    out_fields = base_fields + new_cols
    out_rows = []

    prov_counts: dict[str, dict] = {
        col: {} for col in ["sst_mean_c", "salinity_mean_psu", "ph_mean", "depth_mean_m", "do_mean_mlL"]
    }

    for row in metadata:
        sp      = row["organism"]
        habitat = row.get("habitat_type", "coastal_estuarine")
        v1      = bio_v1.get(sp)
        v2      = env_v2.get(sp)

        # SST: surface is correct - no depth-mean needed
        sst_val, sst_prov = pick(v1, None, "sst_mean_c", "", habitat, "sst_mean_c")

        # Salinity: prefer depth-mean v2, fall back to surface v1
        sal_val, sal_prov = pick(v1, v2, "salinity_mean_psu", "salinity_mean_psu_v2", habitat, "salinity_mean_psu")

        # pH: prefer depth-mean v2, fall back to surface v1
        ph_val, ph_prov = pick(v1, v2, "ph_mean", "ph_mean_v2", habitat, "ph_mean")

        # Depth: prefer observed OBIS occurrence depth, fall back to Bio-ORACLE bathymetry
        dep_val, dep_prov = pick(v1, v2, "depth_mean_m", "depth_mean_m_v2", habitat, "depth_mean_m")

        # DO: surface is fine
        do_val, do_prov = pick(v1, None, "do_mean_mlL", "", habitat, "do_mean_mlL")

        new_data = {
            "sst_mean_c":              sst_val,
            "sst_mean_c_status":       "biooracle_surface" if sst_prov == "biooracle_surface" else sst_prov,
            "sst_mean_c_provenance":   sst_prov,
            "salinity_mean_psu":       sal_val,
            "salinity_mean_psu_status": sal_prov,
            "salinity_mean_psu_provenance": sal_prov,
            "ph_mean":                 ph_val,
            "ph_mean_status":          ph_prov,
            "ph_mean_provenance":      ph_prov,
            "depth_mean_m":            dep_val,
            "depth_mean_m_status":     dep_prov,
            "depth_mean_m_provenance": dep_prov,
            "do_mean_mlL":             do_val,
            "do_mean_mlL_status":      do_prov,
            "do_mean_mlL_provenance":  do_prov,
        }

        for col in prov_counts:
            prov = new_data[f"{col}_provenance"]
            prov_counts[col][prov] = prov_counts[col].get(prov, 0) + 1

        out_rows.append({**row, **new_data})

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nEnriched metadata -> {OUT_CSV}")
    print("\nProvenance breakdown:")
    for col, counts in prov_counts.items():
        parts = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {col:20s}: {parts}")


if __name__ == "__main__":
    main()
