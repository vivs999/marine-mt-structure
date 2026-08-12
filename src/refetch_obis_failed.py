#!/usr/bin/env python3
"""
Re-fetch OBIS occurrences for species that returned 0 records, using
cleaned primary binomial names and scientific synonym fallbacks.

The original fetch sent full UniProt organism strings like
  "Scyliorhinus torazame (Cloudy catshark) (Catulus torazame)"
OBIS only matches clean scientific names, so those all returned 0.

This script:
  1. Reads the existing obis_occurrences.csv to find 0-record species
  2. Parses the clean primary binomial (before first parenthesis)
  3. Tries primary -> synonyms from parentheticals -> GBIF
  4. Merges results back into obis_occurrences.csv and obis_sample_points.csv
"""

from __future__ import annotations
import csv, re, statistics, time
from pathlib import Path

import requests

OCC_CSV = Path("data/processed/obis_occurrences.csv")
PTS_CSV = Path("data/processed/obis_sample_points.csv")

OBIS_API = "https://api.obis.org/v3/occurrence"
GBIF_API = "https://api.gbif.org/v1/occurrence/search"
HEADERS  = {"User-Agent": "mtsef-research/1.0"}

MAX_POINTS = 50
MIN_RECORDS = 3


def parse_names(name: str) -> tuple[str, list[str]]:
    """Return (primary_binomial, [scientific_synonyms])."""
    primary = re.sub(r'\s*\(.*', '', name).strip()
    parens  = re.findall(r'\(([^)]+)\)', name)
    synonyms = [p for p in parens if re.match(r'^[A-Z][a-z]+ [a-z]', p)]
    return primary, synonyms


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
        return [
            rec for rec in r.json().get("results", [])
            if rec.get("decimalLatitude") is not None
            and rec.get("decimalLongitude") is not None
        ]
    except Exception:
        return []


def query_gbif(species: str, limit: int = 200) -> list[dict]:
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
        out = []
        for rec in r.json().get("results", []):
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            if lat is not None and lon is not None:
                out.append({"decimalLatitude": lat, "decimalLongitude": lon,
                            "depth": rec.get("depth")})
        return out
    except Exception:
        return []


def aggregate(records, species, habitat):
    lats   = [r["decimalLatitude"]  for r in records]
    lons   = [r["decimalLongitude"] for r in records]
    depths = [r["depth"] for r in records if r.get("depth") is not None]

    agg = {
        "organism":       species,
        "habitat_type":   habitat,
        "n_obis_records": len(records),
        "lat_mean":   round(statistics.mean(lats), 4),
        "lon_mean":   round(statistics.mean(lons), 4),
        "lat_sd":     round(statistics.stdev(lats), 4) if len(lats) > 1 else 0,
        "depth_median_m": round(statistics.median(depths), 1) if depths else "",
        "depth_p10_m":    round(sorted(depths)[int(len(depths)*0.1)], 1) if depths else "",
        "depth_p90_m":    round(sorted(depths)[int(len(depths)*0.9)], 1) if depths else "",
        "n_depth_records": len(depths),
        "source": "obis",
    }
    step   = max(1, len(records) // MAX_POINTS)
    sample = records[::step][:MAX_POINTS]
    pts    = [{"organism": species, "lat": rec["decimalLatitude"],
               "lon": rec["decimalLongitude"], "depth": rec.get("depth", "")}
              for rec in sample]
    return agg, pts


def main():
    # Load existing occurrences
    occ_rows = list(csv.DictReader(open(OCC_CSV, newline="")))
    pts_rows = list(csv.DictReader(open(PTS_CSV, newline="")))

    failed = [r for r in occ_rows if int(r.get("n_obis_records", 0) or 0) == 0]
    print(f"Species to retry: {len(failed)}")

    recovered = 0
    new_pts: list[dict] = []

    for i, row in enumerate(failed):
        orig_name = row["organism"]
        habitat   = row["habitat_type"]
        primary, synonyms = parse_names(orig_name)

        # Build candidate names to try
        candidates = [primary] + synonyms
        candidates = list(dict.fromkeys(candidates))  # deduplicate, preserve order

        records = []
        source  = "none"
        tried   = []

        for candidate in candidates:
            tried.append(candidate)
            recs = query_obis(candidate)
            time.sleep(0.3)
            if len(recs) >= MIN_RECORDS:
                records = recs
                source  = "obis"
                break
            # Try GBIF for this candidate too
            gbif = query_gbif(candidate)
            time.sleep(0.2)
            if len(gbif) >= MIN_RECORDS:
                records = gbif
                source  = "gbif_fallback"
                break

        if records:
            agg, pts = aggregate(records, orig_name, habitat)
            agg["source"] = source
            # Replace in occ_rows
            for j, existing in enumerate(occ_rows):
                if existing["organism"] == orig_name:
                    occ_rows[j] = agg
                    break
            new_pts.extend(pts)
            recovered += 1
            print(f"[{i+1}/{len(failed)}] RECOVERED {orig_name[:50]} "
                  f"via '{tried[-1]}' ({len(records)} records, src={source})")
        else:
            print(f"[{i+1}/{len(failed)}] still no data: {orig_name[:50]} "
                  f"(tried: {tried})")

    # Write updated files
    occ_fields = [
        "organism", "habitat_type", "n_obis_records",
        "lat_mean", "lon_mean", "lat_sd",
        "depth_median_m", "depth_p10_m", "depth_p90_m",
        "n_depth_records", "source",
    ]
    with open(OCC_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=occ_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(occ_rows)

    all_pts = pts_rows + new_pts
    pt_fields = ["organism", "lat", "lon", "depth"]
    with open(PTS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pt_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_pts)

    still_zero = sum(1 for r in occ_rows if int(r.get("n_obis_records", 0) or 0) == 0)
    print(f"\nRecovered: {recovered}/{len(failed)} species")
    print(f"Still no data: {still_zero} species")
    print(f"New sample points: {len(new_pts)}")
    print(f"Total sample points: {len(all_pts)}")


if __name__ == "__main__":
    main()
