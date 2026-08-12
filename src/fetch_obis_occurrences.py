#!/usr/bin/env python3
"""
Phase 1.1 - OBIS species occurrence fetch.

For each unique species in metadata_populated.csv, query the OBIS REST API
for marine occurrence records and extract:
  - representative lat/lon points (for Bio-ORACLE sampling - internal use only)
  - depth median/range
  - occurrence count

Outputs:
  data/processed/obis_occurrences.csv    -- per-species aggregated stats
  data/processed/obis_sample_points.csv  -- up to 50 lat/lon points per species
"""

from __future__ import annotations
import csv, json, time, statistics
from pathlib import Path
import requests

METADATA_CSV = Path("data/processed/metadata_populated.csv")
OUT_OCC = Path("data/processed/obis_occurrences.csv")
OUT_PTS = Path("data/processed/obis_sample_points.csv")

OBIS_API = "https://api.obis.org/v3/occurrence"
GBIF_API = "https://api.gbif.org/v1/occurrence/search"
HEADERS = {"User-Agent": "mtsef-research/1.0"}

MAX_POINTS_PER_SPECIES = 50
MIN_OBIS_RECORDS = 5


def load_species(path: Path) -> list[dict]:
    seen = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sp = row["organism"]
            if sp not in seen:
                seen[sp] = {
                    "organism": sp,
                    "uniprot_ids": [],
                    "habitat_type": row["habitat_type"],
                }
            seen[sp]["uniprot_ids"].append(row["uniprot_id"])
    return list(seen.values())


def query_obis(species: str, size: int = 500) -> list[dict]:
    params = {
        "scientificname": species,
        "fields": "decimalLatitude,decimalLongitude,depth,date_year",
        "size": size,
        "mof": "false",
    }
    try:
        r = requests.get(OBIS_API, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        records = data.get("results", [])
        return [
            rec for rec in records
            if rec.get("decimalLatitude") is not None
            and rec.get("decimalLongitude") is not None
        ]
    except Exception:
        return []


def query_gbif(species: str, limit: int = 100) -> list[dict]:
    params = {
        "scientificName": species,
        "hasCoordinate": "true",
        "occurrenceStatus": "PRESENT",
        "limit": limit,
    }
    try:
        r = requests.get(GBIF_API, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []
        results = r.json().get("results", [])
        out = []
        for rec in results:
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            if lat is not None and lon is not None:
                out.append({
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "depth": rec.get("depth"),
                })
        return out
    except Exception:
        return []


def aggregate(records: list[dict], species: str, habitat: str) -> tuple[dict, list[dict]]:
    lats = [r["decimalLatitude"] for r in records]
    lons = [r["decimalLongitude"] for r in records]
    depths = [r["depth"] for r in records if r.get("depth") is not None]

    agg = {
        "organism": species,
        "habitat_type": habitat,
        "n_obis_records": len(records),
        "lat_mean": round(statistics.mean(lats), 4),
        "lon_mean": round(statistics.mean(lons), 4),
        "lat_sd": round(statistics.stdev(lats), 4) if len(lats) > 1 else 0,
        "depth_median_m": round(statistics.median(depths), 1) if depths else "",
        "depth_p10_m": round(sorted(depths)[int(len(depths) * 0.1)], 1) if depths else "",
        "depth_p90_m": round(sorted(depths)[int(len(depths) * 0.9)], 1) if depths else "",
        "n_depth_records": len(depths),
        "source": "obis",
    }

    # Sample up to MAX_POINTS_PER_SPECIES representative points
    step = max(1, len(records) // MAX_POINTS_PER_SPECIES)
    sample = records[::step][:MAX_POINTS_PER_SPECIES]
    pts = [
        {
            "organism": species,
            "lat": rec["decimalLatitude"],
            "lon": rec["decimalLongitude"],
            "depth": rec.get("depth", ""),
        }
        for rec in sample
    ]
    return agg, pts


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    species_list = load_species(METADATA_CSV)
    unique_species = [s["organism"] for s in species_list]
    print(f"Species to query: {len(unique_species)}")

    occ_rows = []
    pt_rows = []

    for i, sp_info in enumerate(species_list):
        sp = sp_info["organism"]
        habitat = sp_info["habitat_type"]
        print(f"[{i+1}/{len(species_list)}] {sp}", end=" ... ", flush=True)

        records = query_obis(sp)
        source = "obis"
        time.sleep(0.3)

        if len(records) < MIN_OBIS_RECORDS:
            gbif = query_gbif(sp)
            time.sleep(0.2)
            if gbif:
                records = gbif
                source = "gbif_fallback"

        if not records:
            print(f"no records")
            occ_rows.append({
                "organism": sp,
                "habitat_type": habitat,
                "n_obis_records": 0,
                "lat_mean": "",
                "lon_mean": "",
                "lat_sd": "",
                "depth_median_m": "",
                "depth_p10_m": "",
                "depth_p90_m": "",
                "n_depth_records": 0,
                "source": "none",
            })
            continue

        agg, pts = aggregate(records, sp, habitat)
        agg["source"] = source
        occ_rows.append(agg)
        pt_rows.extend(pts)
        print(f"{len(records)} records, {len(pts)} sample pts, depth_median={agg['depth_median_m']}")

    # Write occurrences
    occ_fields = [
        "organism", "habitat_type", "n_obis_records",
        "lat_mean", "lon_mean", "lat_sd",
        "depth_median_m", "depth_p10_m", "depth_p90_m",
        "n_depth_records", "source",
    ]
    with open(OUT_OCC, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=occ_fields)
        w.writeheader()
        w.writerows(occ_rows)
    print(f"\nOccurrences -> {OUT_OCC}")

    # Write sample points
    pt_fields = ["organism", "lat", "lon", "depth"]
    with open(OUT_PTS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pt_fields)
        w.writeheader()
        w.writerows(pt_rows)
    print(f"Sample points -> {OUT_PTS} ({len(pt_rows)} total points)")

    covered = sum(1 for r in occ_rows if r["n_obis_records"] > 0)
    print(f"\nCoverage: {covered}/{len(occ_rows)} species with occurrence data")


if __name__ == "__main__":
    main()
