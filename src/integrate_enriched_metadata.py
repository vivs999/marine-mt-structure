#!/usr/bin/env python3
"""
Phase 1.3 - Merge OBIS depth + Bio-ORACLE environmental layers into metadata.

Provenance hierarchy per field:
  1. observed (existing real values in metadata_populated.csv)
  2. biooracle_sampled (from Bio-ORACLE via OBIS occurrence points)
  3. obis_aggregate (depth from OBIS records)
  4. habitat_default (fallback imputed value)

Output: data/processed/metadata_enriched_v2.csv
"""

from __future__ import annotations
import csv
from pathlib import Path

METADATA_CSV  = Path("data/processed/metadata_populated.csv")
OCC_CSV       = Path("data/processed/obis_occurrences.csv")
BIO_CSV       = Path("data/processed/biooracle_environmental.csv")
OUT_CSV       = Path("data/processed/metadata_enriched_v2.csv")

HABITAT_DEFAULTS = {
    "coastal_estuarine": {
        "temperature_c": 22.0,
        "ph":            8.1,
        "salinity_psu":  32.0,
        "depth_m":       30.0,
        "sst_mean_c":    22.0,
        "salinity_mean_psu": 32.0,
        "ph_mean":       8.1,
        "depth_mean_m":  30.0,
        "do_mean_mlL":   6.5,
    },
    "open_ocean": {
        "temperature_c": 15.0,
        "ph":            8.1,
        "salinity_psu":  35.0,
        "depth_m":       200.0,
        "sst_mean_c":    15.0,
        "salinity_mean_psu": 35.0,
        "ph_mean":       8.1,
        "depth_mean_m":  200.0,
        "do_mean_mlL":   7.0,
    },
    "polar": {
        "temperature_c": -1.5,
        "ph":            8.0,
        "salinity_psu":  34.0,
        "depth_m":       100.0,
        "sst_mean_c":    -1.5,
        "salinity_mean_psu": 34.0,
        "ph_mean":       8.0,
        "depth_mean_m":  100.0,
        "do_mean_mlL":   8.5,
    },
    "hydrothermal_vent": {
        "temperature_c": 4.0,
        "ph":            6.5,
        "salinity_psu":  35.0,
        "depth_m":       2500.0,
        "sst_mean_c":    4.0,
        "salinity_mean_psu": 35.0,
        "ph_mean":       6.5,
        "depth_mean_m":  2500.0,
        "do_mean_mlL":   3.0,
    },
}


def load_csv_by_organism(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            org = row.get("organism", "")
            if org:
                out[org] = row
    return out


def fill_field(row: dict, bio_row: dict | None, occ_row: dict | None,
               field: str, bio_key: str, occ_key: str | None,
               default: float | None) -> tuple[str, str, str]:
    """Return (value, status, provenance) for a single field."""
    # 1. existing observed
    existing_val = row.get(field, "")
    existing_status = row.get(f"{field}_status", "")
    if existing_val and existing_status == "observed":
        return existing_val, "observed", row.get(f"{field}_provenance", "existing")

    # 2. Bio-ORACLE
    if bio_row and bio_key and bio_row.get(bio_key, ""):
        return bio_row[bio_key], "biooracle_sampled", f"biooracle:{bio_key}"

    # 3. OBIS aggregate (depth only)
    if occ_row and occ_key and occ_row.get(occ_key, ""):
        return occ_row[occ_key], "obis_aggregate", f"obis:{occ_key}"

    # 4. habitat default
    if default is not None:
        return str(default), "habitat_default", f"habitat_default:{row['habitat_type']}"

    return "", "missing", "unavailable"


def main():
    metadata = []
    with open(METADATA_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        base_fields = list(reader.fieldnames)
        metadata = list(reader)

    occ_by_sp = load_csv_by_organism(OCC_CSV)
    bio_by_sp = load_csv_by_organism(BIO_CSV)

    print(f"Metadata rows: {len(metadata)}")
    print(f"OBIS species:  {len(occ_by_sp)}")
    print(f"Bio-ORACLE:    {len(bio_by_sp)}")

    # New target columns (ML regression targets from Bio-ORACLE)
    new_cols = [
        "sst_mean_c", "sst_mean_c_status", "sst_mean_c_provenance",
        "salinity_mean_psu", "salinity_mean_psu_status", "salinity_mean_psu_provenance",
        "ph_mean", "ph_mean_status", "ph_mean_provenance",
        "depth_mean_m", "depth_mean_m_status", "depth_mean_m_provenance",
        "do_mean_mlL", "do_mean_mlL_status", "do_mean_mlL_provenance",
    ]

    out_fields = base_fields + new_cols
    out_rows = []

    stats = {
        "sst_observed": 0, "sst_biooracle": 0, "sst_default": 0,
        "sal_biooracle": 0, "sal_default": 0,
        "ph_biooracle": 0, "ph_default": 0,
        "depth_biooracle": 0, "depth_obis": 0, "depth_default": 0,
        "do_biooracle": 0, "do_default": 0,
    }

    for row in metadata:
        sp = row["organism"]
        habitat = row.get("habitat_type", "coastal_estuarine")
        bio = bio_by_sp.get(sp)
        occ = occ_by_sp.get(sp)
        defs = HABITAT_DEFAULTS.get(habitat, HABITAT_DEFAULTS["coastal_estuarine"])

        # Enrich existing temperature_c / ph / salinity_psu with Bio-ORACLE if missing
        if bio:
            if not row.get("temperature_c") or row.get("temperature_c_status") != "observed":
                if bio.get("sst_mean_c"):
                    row["temperature_c"] = bio["sst_mean_c"]
                    row["temperature_c_status"] = "biooracle_sampled"
                    row["temperature_c_provenance"] = "biooracle:BO22_tempmean_ss"
            if not row.get("ph") or row.get("ph_status") != "observed":
                if bio.get("ph_mean"):
                    row["ph"] = bio["ph_mean"]
                    row["ph_status"] = "biooracle_sampled"
                    row["ph_provenance"] = "biooracle:BO22_phmean_ss"
            if not row.get("salinity_psu") or row.get("salinity_psu_status") != "observed":
                if bio.get("salinity_mean_psu"):
                    row["salinity_psu"] = bio["salinity_mean_psu"]
                    row["salinity_psu_status"] = "biooracle_sampled"
                    row["salinity_psu_provenance"] = "biooracle:BO22_salinitymean_ss"
            if not row.get("depth_m") or row.get("depth_m_status") != "observed":
                if occ and occ.get("depth_median_m"):
                    row["depth_m"] = occ["depth_median_m"]
                    row["depth_m_status"] = "obis_aggregate"
                    row["depth_m_provenance"] = "obis:depth_median"
                elif bio.get("depth_mean_m"):
                    row["depth_m"] = bio["depth_mean_m"]
                    row["depth_m_status"] = "biooracle_sampled"
                    row["depth_m_provenance"] = "biooracle:BO_bathymean"

        # Build new ML target columns
        FIELD_MAP = [
            # (out_col, bio_key, occ_key, default_key)
            ("sst_mean_c",        "sst_mean_c",        None,            "sst_mean_c"),
            ("salinity_mean_psu", "salinity_mean_psu", None,            "salinity_mean_psu"),
            ("ph_mean",           "ph_mean",           None,            "ph_mean"),
            ("depth_mean_m",      "depth_mean_m",      "depth_median_m","depth_mean_m"),
            ("do_mean_mlL",       "do_mean_mlL",       None,            "do_mean_mlL"),
        ]

        new_data = {}
        for out_col, bio_key, occ_key, def_key in FIELD_MAP:
            bio_val = bio.get(bio_key, "") if bio else ""
            occ_val = occ.get(occ_key, "") if (occ and occ_key) else ""
            default = defs.get(def_key)

            if bio_val:
                val, status, prov = bio_val, "biooracle_sampled", f"biooracle:{bio_key}"
            elif occ_val:
                val, status, prov = occ_val, "obis_aggregate", f"obis:{occ_key}"
            elif default is not None:
                val, status, prov = str(default), "habitat_default", f"habitat_default:{habitat}"
            else:
                val, status, prov = "", "missing", "unavailable"

            new_data[out_col] = val
            new_data[f"{out_col}_status"] = status
            new_data[f"{out_col}_provenance"] = prov

        out_row = {**row, **new_data}
        out_rows.append(out_row)

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nEnriched metadata -> {OUT_CSV}")
    # Coverage summary
    for col in ["sst_mean_c", "salinity_mean_psu", "ph_mean", "depth_mean_m", "do_mean_mlL"]:
        n_biooracle = sum(1 for r in out_rows if r.get(f"{col}_status") == "biooracle_sampled")
        n_default   = sum(1 for r in out_rows if r.get(f"{col}_status") == "habitat_default")
        n_missing   = sum(1 for r in out_rows if r.get(f"{col}_status") in ("missing",""))
        print(f"  {col:20s}: biooracle={n_biooracle}, default={n_default}, missing={n_missing}")


if __name__ == "__main__":
    main()
