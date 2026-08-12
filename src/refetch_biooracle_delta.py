#!/usr/bin/env python3
"""
Bio-ORACLE delta sampling - only queries species NOT already in
biooracle_environmental.csv. Merges results into that file.

Run after refetch_obis_failed.py to fill in newly recovered species.
"""

from __future__ import annotations
import csv, statistics, time
from pathlib import Path

import requests

OCC_CSV        = Path("data/processed/obis_occurrences.csv")
BIO_CSV        = Path("data/processed/biooracle_environmental.csv")

ERDDAP_BASE   = "https://erddap.bio-oracle.org/erddap/griddap"
HEADERS       = {"User-Agent": "mtsef-research/1.0"}
REQUEST_DELAY = 0.15
MAX_RETRIES   = 2

LAYERS = {
    "sst_mean_c":        ("thetao_baseline_2000_2019_depthsurf", "thetao_mean", "2000-01-01"),
    "sst_min_c":         ("thetao_baseline_2000_2019_depthsurf", "thetao_min",  "2000-01-01"),
    "sst_max_c":         ("thetao_baseline_2000_2019_depthsurf", "thetao_max",  "2000-01-01"),
    "salinity_mean_psu": ("so_baseline_2000_2019_depthsurf",    "so_mean",     "2000-01-01"),
    "salinity_min_psu":  ("so_baseline_2000_2019_depthsurf",    "so_min",      "2000-01-01"),
    "salinity_max_psu":  ("so_baseline_2000_2019_depthsurf",    "so_max",      "2000-01-01"),
    "ph_mean":           ("ph_baseline_2000_2018_depthsurf",    "ph_mean",     "2000-01-01"),
    "do_mean_mlL":       ("o2_baseline_2000_2018_depthsurf",    "o2_mean",     "2000-01-01"),
    "depth_mean_m":      ("terrain_characteristics",            "bathymetry_mean", "1970-01-01"),
}


def query_point(dataset_id, var_name, lat, lon, time_coord):
    if lon > 180:
        lon -= 360
    url = (f"{ERDDAP_BASE}/{dataset_id}.csv"
           f"?{var_name}[({time_coord})][({lat:.4f})][({lon:.4f})]")
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
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
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5)
    return None


def sample_species(sp, lat, lon):
    """Sample all layers at the species' mean occurrence centroid (1 point)."""
    row: dict = {"organism": sp, "n_sample_points": 1}
    for col_name, (dataset_id, var_name, time_coord) in LAYERS.items():
        val = query_point(dataset_id, var_name, lat, lon, time_coord)
        time.sleep(REQUEST_DELAY)
        if val is not None:
            if col_name == "depth_mean_m":
                val = abs(val)
            elif col_name == "do_mean_mlL":
                val = val * 0.022391
            row[col_name] = round(val, 4)
            row[f"{col_name}_n"] = 1
        else:
            row[col_name] = ""
            row[f"{col_name}_n"] = 0
    return row


def main():
    # Load existing Bio-ORACLE coverage
    existing_bio: dict[str, dict] = {}
    if BIO_CSV.exists():
        existing_bio = {r["organism"]: r for r in csv.DictReader(open(BIO_CSV, newline=""))}
    print(f"Already sampled: {len(existing_bio)} species")

    # Load OBIS occurrences - use mean lat/lon centroid, skip already-sampled species
    centroids: dict[str, tuple[float, float]] = {}
    with open(OCC_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            sp = row["organism"]
            if sp in existing_bio:
                continue
            try:
                lat = float(row["lat_mean"])
                lon = float(row["lon_mean"])
                if row.get("n_obis_records", "0") != "0":
                    centroids[sp] = (lat, lon)
            except (ValueError, KeyError):
                pass

    if not centroids:
        print("No new species to sample - nothing to do.")
        return

    print(f"New species to sample: {len(centroids)} (centroid mode - 1 pt x 9 layers each)\n")

    fields = ["organism", "n_sample_points"]
    for col_name in LAYERS:
        fields.extend([col_name, f"{col_name}_n"])

    # Write incrementally - append each species as it completes
    # Start by rewriting the file with existing rows, then append
    all_rows = list(existing_bio.values())
    with open(BIO_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    new_count = 0
    for i, (sp, (lat, lon)) in enumerate(centroids.items()):
        print(f"[{i+1}/{len(centroids)}] {sp[:50]} ...", end=" ", flush=True)
        row = sample_species(sp, lat, lon)
        covered = sum(1 for k in LAYERS if row.get(k, "") != "")
        print(f"{covered}/{len(LAYERS)} layers", flush=True)

        # Append immediately so progress survives a kill
        with open(BIO_CSV, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writerow(row)
        new_count += 1

    total = len(existing_bio) + new_count
    print(f"\nAdded {new_count} new species. Total: {total}")
    print(f"Updated -> {BIO_CSV}")


if __name__ == "__main__":
    main()
