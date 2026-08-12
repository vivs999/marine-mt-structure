#!/usr/bin/env python3
"""
Phase 1.4 - High-quality environmental label enrichment (v2).

Improvements over v1:
  1. Depth: use actual OBIS occurrence depths (observed, not inferred bathymetry)
  2. Salinity/pH: query Bio-ORACLE depth-mean layers instead of surface-only

Outputs: data/processed/environmental_labels_v2.csv
"""

from __future__ import annotations
import csv, statistics, time
from pathlib import Path

import requests

SAMPLE_PTS_CSV = Path("data/processed/obis_sample_points.csv")
BIO_V1_CSV     = Path("data/processed/biooracle_environmental.csv")
OUT_CSV        = Path("data/processed/environmental_labels_v2.csv")

ERDDAP_BASE = "https://erddap.bio-oracle.org/erddap/griddap"
HEADERS     = {"User-Agent": "mtsef-research/1.0"}
REQUEST_DELAY = 0.3

# Depth-mean layers: integrate over the water column rather than surface snapshot.
# This differentiates deep benthic MTs from shallow coastal ones.
DEPTHMEAN_LAYERS = {
    "salinity_mean_psu_v2": ("so_baseline_2000_2019_depthmean",  "so_mean",  "2000-01-01"),
    "ph_mean_v2":           ("ph_baseline_2000_2018_depthmean",  "ph_mean",  "2000-01-01"),
}


def load_sample_points() -> dict[str, dict]:
    """Return per-species lat/lon list and observed depth list."""
    species: dict[str, dict] = {}
    with open(SAMPLE_PTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            sp = row["organism"]
            if sp not in species:
                species[sp] = {"latlons": [], "depths": []}
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                species[sp]["latlons"].append((lat, lon))
            except (ValueError, KeyError):
                continue
            depth_str = row.get("depth", "").strip()
            if depth_str:
                try:
                    d = float(depth_str)
                    if d >= 0:
                        species[sp]["depths"].append(d)
                except ValueError:
                    pass
    return species


def load_biooracle_v1() -> dict[str, dict]:
    """Load existing Bio-ORACLE v1 data (keeps SST, DO, bathymetry depth)."""
    out: dict[str, dict] = {}
    if not BIO_V1_CSV.exists():
        return out
    with open(BIO_V1_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            org = row.get("organism", "")
            if org:
                out[org] = row
    return out


def verify_dataset(dataset_id: str) -> bool:
    url = f"{ERDDAP_BASE}/{dataset_id}.das"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        return r.status_code == 200
    except Exception:
        return False


def query_point(dataset_id: str, var_name: str, lat: float, lon: float,
                time_coord: str) -> float | None:
    if lon > 180:
        lon -= 360
    url = (
        f"{ERDDAP_BASE}/{dataset_id}.csv"
        f"?{var_name}[({time_coord})][({lat:.4f})][({lon:.4f})]"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                return None
            lines = r.text.strip().splitlines()
            if len(lines) < 3:
                return None
            val_str = lines[2].split(",")[-1].strip()
            if val_str in ("", "NaN"):
                return None
            return float(val_str)
        except Exception:
            if attempt < 2:
                time.sleep(1.0)
    return None


def sample_depthmean(latlons: list[tuple[float, float]],
                     active: dict[str, tuple[str, str, str]]) -> dict[str, float | None]:
    step = max(1, len(latlons) // 15)
    pts = latlons[::step][:15]

    results: dict[str, float | None] = {col: None for col in active}
    for col, (dataset_id, var_name, time_coord) in active.items():
        vals: list[float] = []
        for lat, lon in pts:
            v = query_point(dataset_id, var_name, lat, lon, time_coord)
            time.sleep(REQUEST_DELAY)
            if v is not None:
                vals.append(v)
        if vals:
            results[col] = round(statistics.median(vals), 4)
    return results


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Loading sample points...")
    sp_data = load_sample_points()
    print(f"  {len(sp_data)} species, "
          f"{sum(len(v['depths']) for v in sp_data.values())} depth observations")

    print("Loading Bio-ORACLE v1 data...")
    bio_v1 = load_biooracle_v1()
    print(f"  {len(bio_v1)} species")

    print("\nVerifying depth-mean ERDDAP datasets...")
    active: dict[str, tuple[str, str, str]] = {}
    for col, (ds, var, t) in DEPTHMEAN_LAYERS.items():
        ok = verify_dataset(ds)
        print(f"  {col:30s} -> {ds} [{'OK' if ok else 'SKIP'}]")
        if ok:
            active[col] = (ds, var, t)
        time.sleep(0.1)

    rows = []
    species_list = sorted(set(list(sp_data.keys()) + list(bio_v1.keys())))
    print(f"\nProcessing {len(species_list)} species...")

    for i, sp in enumerate(species_list):
        sp_info = sp_data.get(sp, {"latlons": [], "depths": []})
        v1 = bio_v1.get(sp, {})
        row: dict = {"organism": sp}

        # Depth (observed occurrence depth, not bathymetry)
        depths = sp_info["depths"]
        if depths:
            row["depth_mean_m_v2"]      = round(statistics.median(depths), 1)
            row["depth_mean_m_v2_n"]    = len(depths)
            row["depth_mean_m_v2_src"]  = "obis_occurrence_depth"
        else:
            row["depth_mean_m_v2"]      = ""
            row["depth_mean_m_v2_n"]    = 0
            row["depth_mean_m_v2_src"]  = "unavailable"

        # SST and DO: keep v1 values (surface layers are appropriate)
        for passthrough in ["sst_mean_c", "sst_min_c", "sst_max_c",
                            "salinity_min_psu", "salinity_max_psu", "do_mean_mlL"]:
            row[passthrough] = v1.get(passthrough, "")

        # Salinity/pH: depth-mean layers
        if active and sp_info["latlons"]:
            print(f"  [{i+1}/{len(species_list)}] {sp} ({len(sp_info['latlons'])} pts, "
                  f"{len(depths)} depth obs) ...", end=" ", flush=True)
            dm = sample_depthmean(sp_info["latlons"], active)
            for col, val in dm.items():
                row[col] = val if val is not None else ""
            covered = sum(1 for v in dm.values() if v is not None)
            print(f"{covered}/{len(active)} layers")
        else:
            for col in active:
                row[col] = v1.get(col.replace("_v2", ""), "")
            if not active:
                print(f"  [{i+1}/{len(species_list)}] {sp} (no ERDDAP, using v1)")

        rows.append(row)

    # Write output
    fields = (
        ["organism", "depth_mean_m_v2", "depth_mean_m_v2_n", "depth_mean_m_v2_src"]
        + ["sst_mean_c", "sst_min_c", "sst_max_c",
           "salinity_min_psu", "salinity_max_psu", "do_mean_mlL"]
        + list(active.keys())
    )
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Coverage summary
    print(f"\nOutput -> {OUT_CSV}")
    n_depth = sum(1 for r in rows if r.get("depth_mean_m_v2"))
    print(f"  depth_mean_m_v2:   {n_depth}/{len(rows)} species with observed depth")
    for col in active:
        n = sum(1 for r in rows if r.get(col))
        print(f"  {col:30s}: {n}/{len(rows)} species")


if __name__ == "__main__":
    main()
