#!/usr/bin/env python3
"""
Phase 1.2 - Bio-ORACLE environmental layer sampling via ERDDAP REST API.

Reads data/processed/obis_sample_points.csv and queries Bio-ORACLE v3
gridded ocean layers at each species' occurrence coordinates via the
ERDDAP griddap REST API. Aggregates to per-species medians.

No extra packages needed - uses only requests (already in requirements.txt).

Bio-ORACLE v3 ERDDAP: https://erddap.bio-oracle.org/erddap/griddap/

Output: data/processed/biooracle_environmental.csv
"""

from __future__ import annotations
import csv, json, statistics, time
from pathlib import Path

import requests

SAMPLE_PTS_CSV = Path("data/processed/obis_sample_points.csv")
OUT_CSV = Path("data/processed/biooracle_environmental.csv")

ERDDAP_BASE = "https://erddap.bio-oracle.org/erddap/griddap"
HEADERS = {"User-Agent": "mtsef-research/1.0"}
REQUEST_DELAY = 0.25  # seconds between ERDDAP requests

# Bio-ORACLE v3 dataset IDs, variable names, and the time coordinate to use.
# ERDDAP griddap requires all 3 dimensions: [(time)][(lat)][(lon)]
# Format: output_col_name -> (dataset_id, variable_name, time_coord)
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

# Seconds to wait and retry count for transient ERDDAP errors
MAX_RETRIES = 3


def load_sample_points() -> dict[str, list[tuple[float, float]]]:
    pts: dict[str, list[tuple[float, float]]] = {}
    with open(SAMPLE_PTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            sp = row["organism"]
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (ValueError, KeyError):
                continue
            pts.setdefault(sp, []).append((lat, lon))
    return pts


def verify_dataset(dataset_id: str) -> bool:
    """Quick check that the dataset exists on ERDDAP."""
    url = f"{ERDDAP_BASE}/{dataset_id}.das"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        return r.status_code == 200
    except Exception:
        return False


def query_point(
    dataset_id: str, var_name: str, lat: float, lon: float, time_coord: str
) -> float | None:
    """
    Query a single lat/lon from a Bio-ORACLE ERDDAP griddap dataset.
    All Bio-ORACLE v3 datasets have 3 dimensions: [time][latitude][longitude].
    ERDDAP nearest-neighbor syntax: ?var[(time)][(lat)][(lon)]
    """
    # Clamp lon to [-180, 180]
    if lon > 180:
        lon -= 360

    url = (
        f"{ERDDAP_BASE}/{dataset_id}.csv"
        f"?{var_name}[({time_coord})][({lat:.4f})][({lon:.4f})]"
    )

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                return None
            lines = r.text.strip().splitlines()
            # CSV format: header row, units row, data row(s)
            if len(lines) < 3:
                return None
            data_line = lines[2]  # first data row (after header + units)
            parts = data_line.split(",")
            # Variable value is the last column
            val_str = parts[-1].strip()
            if val_str in ("", "NaN", "NaN,NaN"):
                return None
            return float(val_str)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.0)
    return None


def sample_species(
    sp: str,
    points: list[tuple[float, float]],
    active_layers: dict[str, tuple[str, str, str]],
) -> dict:
    """Sample all active layers for one species. Points are subsampled to save API calls."""
    # Use at most 10 representative points to limit API calls
    step = max(1, len(points) // 10)
    sample = points[::step][:10]

    row: dict[str, object] = {"organism": sp, "n_sample_points": len(points)}

    for col_name, (dataset_id, var_name, time_coord) in active_layers.items():
        vals: list[float] = []
        for lat, lon in sample:
            val = query_point(dataset_id, var_name, lat, lon, time_coord)
            time.sleep(REQUEST_DELAY)
            if val is not None:
                vals.append(val)

        if vals:
            med = statistics.median(vals)
            # Depth: bathymetry is negative below sea level - store as positive meters
            if col_name == "depth_mean_m":
                med = abs(med)
            # DO: Bio-ORACLE v3 reports mmol/m³; convert to mL/L (1 mmol/m³ = 0.022391 mL/L)
            elif col_name == "do_mean_mlL":
                med = med * 0.022391
            row[col_name] = round(med, 4)
            row[f"{col_name}_n"] = len(vals)
        else:
            row[col_name] = ""
            row[f"{col_name}_n"] = 0

    return row


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    if not SAMPLE_PTS_CSV.exists():
        print(f"ERROR: {SAMPLE_PTS_CSV} not found. Run fetch_obis_occurrences.py first.")
        return

    pts_by_species = load_sample_points()
    if not pts_by_species:
        print(f"No sample points found in {SAMPLE_PTS_CSV}.")
        return

    print(f"Species to sample: {len(pts_by_species)}")
    print(f"Layers requested: {len(LAYERS)}\n")

    # Verify each dataset exists before running the full loop
    print("Verifying Bio-ORACLE ERDDAP datasets...")
    active_layers: dict[str, tuple[str, str, str]] = {}
    for col_name, (dataset_id, var_name, time_coord) in LAYERS.items():
        ok = verify_dataset(dataset_id)
        status = "OK" if ok else "SKIP (not found)"
        print(f"  {col_name:20s} -> {dataset_id} [{status}]")
        if ok:
            active_layers[col_name] = (dataset_id, var_name, time_coord)
        time.sleep(0.1)

    if not active_layers:
        print("\nERROR: No Bio-ORACLE datasets reachable. Check network connectivity.")
        print("If Bio-ORACLE v3 layer IDs have changed, update the LAYERS dict in this script.")
        return

    print(f"\nActive layers: {len(active_layers)}/{len(LAYERS)}\n")

    results = []
    for i, (sp, points) in enumerate(pts_by_species.items()):
        print(f"[{i+1}/{len(pts_by_species)}] {sp} ({len(points)} pts) ...", end=" ", flush=True)
        row = sample_species(sp, points, active_layers)
        results.append(row)
        covered = sum(1 for k in active_layers if row.get(k, "") != "")
        print(f"{covered}/{len(active_layers)} layers")

    # Write output
    fields = ["organism", "n_sample_points"]
    for col_name in LAYERS:
        if col_name in active_layers:
            fields.extend([col_name, f"{col_name}_n"])

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    n_full = sum(
        1 for r in results
        if all(r.get(c, "") != "" for c in active_layers)
    )
    print(f"\nBio-ORACLE data -> {OUT_CSV}")
    print(f"Full coverage (all {len(active_layers)} active layers): {n_full}/{len(results)} species")


if __name__ == "__main__":
    main()
