#!/usr/bin/env python3
"""
Extract metal coordination geometry features from Boltz-predicted Zn²⁺-bound MT structures.

Features capture the actual metal-binding geometry - not apo random coil.

Usage:
  uv run --python 3.12 --with boltz python3 src/extract_metalbound_features.py

Input:  data/raw/pdb_structures_metalbound/{uid}/**/*.cif
Output: data/processed/metalbound_features.csv
"""

from __future__ import annotations
import csv
import math
import statistics
from pathlib import Path

import numpy as np

STRUCT_DIR = Path("data/raw/pdb_structures_metalbound")
OUT_CSV    = Path("data/processed/metalbound_features.csv")

# Zn²⁺ coordination bond length thresholds (Å)
ZN_S_BOND_MAX = 2.8    # Zn-S distance cutoff (crystal avg ~2.33 Å)
ZN_S_BOND_MIN = 1.8


def find_cif_files() -> list[tuple[str, Path]]:
    found = []
    for uid_dir in sorted(STRUCT_DIR.iterdir()):
        if not uid_dir.is_dir():
            continue
        uid = uid_dir.name
        cif = uid_dir / f"boltz_results_{uid}" / "predictions" / uid / f"{uid}_model_0.cif"
        if cif.exists():
            found.append((uid, cif))
    return found


def parse_cif_atoms(cif_text: str) -> list[dict]:
    """
    Parse _atom_site records from mmCIF. Returns list of atom dicts with keys:
      type_symbol, label_atom_id, label_comp_id, label_asym_id,
      label_seq_id, Cartn_x, Cartn_y, Cartn_z
    """
    atoms = []
    in_atom_site = False
    columns: list[str] = []
    col_map: dict[str, int] = {}

    for line in cif_text.splitlines():
        line = line.strip()
        if line == "_atom_site.group_PDB" or line.startswith("_atom_site."):
            in_atom_site = True
            col_name = line.removeprefix("_atom_site.").strip()
            columns.append(col_name)
            col_map[col_name] = len(columns) - 1
            continue
        if not in_atom_site:
            continue
        if line.startswith("_") or line.startswith("loop_") or line.startswith("#"):
            in_atom_site = False
            columns = []
            col_map = {}
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < len(columns):
            continue
        try:
            atom = {col: parts[col_map[col]] for col in col_map}
            atom["x"] = float(atom.get("Cartn_x", "0"))
            atom["y"] = float(atom.get("Cartn_y", "0"))
            atom["z"] = float(atom.get("Cartn_z", "0"))
            atoms.append(atom)
        except (KeyError, ValueError):
            continue
    return atoms


def dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)


def angle_deg(a: dict, vertex: dict, c: dict) -> float:
    """Angle a-vertex-c in degrees."""
    v1 = np.array([a["x"]-vertex["x"], a["y"]-vertex["y"], a["z"]-vertex["z"]])
    v2 = np.array([c["x"]-vertex["x"], c["y"]-vertex["y"], c["z"]-vertex["z"]])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_a)))))


def extract_features(uid: str, cif_path: Path) -> dict | None:
    try:
        atoms = parse_cif_atoms(cif_path.read_text())
    except Exception as e:
        print(f"  CIF parse error {uid}: {e}")
        return None

    if not atoms:
        return None

    # Separate Zn ions and Cys sulfur atoms (SG)
    zn_atoms  = [a for a in atoms if a.get("type_symbol", "").upper() == "ZN"]
    sg_atoms  = [a for a in atoms
                 if a.get("label_atom_id", "") == "SG"
                 and a.get("label_comp_id", "") == "CYS"]
    ca_atoms  = [a for a in atoms if a.get("label_atom_id", "") == "CA"]

    n_zn = len(zn_atoms)
    n_cys_sg = len(sg_atoms)

    if n_zn == 0:
        print(f"  WARNING {uid}: no Zn atoms found in CIF")
        return None

    # Zn-S coordination distances
    zn_s_pairs: list[tuple[dict, dict, float]] = []
    for zn in zn_atoms:
        for sg in sg_atoms:
            d = dist(zn, sg)
            if ZN_S_BOND_MIN <= d <= ZN_S_BOND_MAX:
                zn_s_pairs.append((zn, sg, d))

    zn_s_distances = [d for _, _, d in zn_s_pairs]

    if not zn_s_distances:
        # Boltz might not have placed Zn close to Cys - fall back to nearest distances
        all_dists = [dist(zn, sg) for zn in zn_atoms for sg in sg_atoms]
        zn_s_distances = sorted(all_dists)[:n_zn * 4]  # up to 4 per Zn

    mean_zn_s_dist  = statistics.mean(zn_s_distances) if zn_s_distances else float("nan")
    std_zn_s_dist   = statistics.stdev(zn_s_distances) if len(zn_s_distances) > 1 else 0.0
    n_coordinating  = len(zn_s_distances)
    coord_per_zn    = n_coordinating / n_zn if n_zn > 0 else 0.0

    # S-Zn-S angles (tetrahedral geometry check)
    sznS_angles: list[float] = []
    # Group sulfurs by which Zn they coordinate
    zn_coord_sg: dict[int, list[dict]] = {i: [] for i in range(n_zn)}
    for zn_i, zn in enumerate(zn_atoms):
        for sg in sg_atoms:
            d = dist(zn, sg)
            if ZN_S_BOND_MIN <= d <= ZN_S_BOND_MAX:
                zn_coord_sg[zn_i].append(sg)

    for zn_i, zn in enumerate(zn_atoms):
        sgs = zn_coord_sg[zn_i]
        for j in range(len(sgs)):
            for k in range(j + 1, len(sgs)):
                ang = angle_deg(sgs[j], zn, sgs[k])
                sznS_angles.append(ang)

    # Ideal tetrahedral angle = 109.47°
    mean_szns_angle = statistics.mean(sznS_angles) if sznS_angles else float("nan")
    tetrahedral_deviation = abs(mean_szns_angle - 109.47) if sznS_angles else float("nan")

    # Zn-Zn distances (cluster compactness)
    zn_zn_dists: list[float] = []
    for i in range(len(zn_atoms)):
        for j in range(i + 1, len(zn_atoms)):
            zn_zn_dists.append(dist(zn_atoms[i], zn_atoms[j]))

    mean_zn_zn_dist = statistics.mean(zn_zn_dists) if zn_zn_dists else float("nan")
    min_zn_zn_dist  = min(zn_zn_dists) if zn_zn_dists else float("nan")

    # Metal cluster centre of mass
    zn_x = [a["x"] for a in zn_atoms]
    zn_y = [a["y"] for a in zn_atoms]
    zn_z = [a["z"] for a in zn_atoms]
    com_zn = (statistics.mean(zn_x), statistics.mean(zn_y), statistics.mean(zn_z))

    # Radius of gyration of Zn cluster
    rg_zn = math.sqrt(statistics.mean(
        (x - com_zn[0])**2 + (y - com_zn[1])**2 + (z - com_zn[2])**2
        for x, y, z in zip(zn_x, zn_y, zn_z)
    )) if n_zn > 1 else 0.0

    # Protein Cα radius of gyration
    ca_x = [a["x"] for a in ca_atoms]
    ca_y = [a["y"] for a in ca_atoms]
    ca_z = [a["z"] for a in ca_atoms]
    n_res = len(ca_atoms)
    if n_res > 1:
        com_ca = (statistics.mean(ca_x), statistics.mean(ca_y), statistics.mean(ca_z))
        rg_protein = math.sqrt(statistics.mean(
            (x - com_ca[0])**2 + (y - com_ca[1])**2 + (z - com_ca[2])**2
            for x, y, z in zip(ca_x, ca_y, ca_z)
        ))
    else:
        rg_protein = float("nan")

    # Distance from protein CoM to Zn cluster CoM
    if n_res > 1:
        metal_protein_offset = math.sqrt(
            (com_zn[0] - com_ca[0])**2 +
            (com_zn[1] - com_ca[1])**2 +
            (com_zn[2] - com_ca[2])**2
        )
    else:
        metal_protein_offset = float("nan")

    return {
        "uniprot_id":           uid,
        "mb_n_zn":              n_zn,
        "mb_n_cys_coordinating": n_coordinating,
        "mb_coord_per_zn":      round(coord_per_zn, 3),
        "mb_mean_zn_s_dist":    round(mean_zn_s_dist, 4),
        "mb_std_zn_s_dist":     round(std_zn_s_dist, 4),
        "mb_mean_szns_angle":   round(mean_szns_angle, 3),
        "mb_tetrahedral_dev":   round(tetrahedral_deviation, 3),
        "mb_mean_zn_zn_dist":   round(mean_zn_zn_dist, 4),
        "mb_min_zn_zn_dist":    round(min_zn_zn_dist, 4),
        "mb_rg_zn_cluster":     round(rg_zn, 4),
        "mb_rg_protein":        round(rg_protein, 4),
        "mb_metal_protein_offset": round(metal_protein_offset, 4),
        "mb_n_residues":        n_res,
    }


FIELDS = [
    "uniprot_id",
    "mb_n_zn", "mb_n_cys_coordinating", "mb_coord_per_zn",
    "mb_mean_zn_s_dist", "mb_std_zn_s_dist",
    "mb_mean_szns_angle", "mb_tetrahedral_dev",
    "mb_mean_zn_zn_dist", "mb_min_zn_zn_dist",
    "mb_rg_zn_cluster", "mb_rg_protein", "mb_metal_protein_offset",
    "mb_n_residues",
]


def main():
    cif_files = find_cif_files()
    print(f"CIF files found: {len(cif_files)}")

    if not cif_files:
        print(f"No CIF files under {STRUCT_DIR}. Run predict_metal_bound_structures.py first.")
        return

    rows = []
    for i, (uid, cif_path) in enumerate(cif_files):
        print(f"[{i+1}/{len(cif_files)}] {uid}", end=" ... ", flush=True)
        feat = extract_features(uid, cif_path)
        if feat:
            rows.append(feat)
            print(f"Zn={feat['mb_n_zn']}  coord/Zn={feat['mb_coord_per_zn']}  "
                  f"Zn-S={feat['mb_mean_zn_s_dist']:.2f}Å  "
                  f"∠S-Zn-S={feat['mb_mean_szns_angle']:.1f}°")
        else:
            print("SKIP")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nExtracted features for {len(rows)}/{len(cif_files)} structures -> {OUT_CSV}")

    if rows:
        coord_vals = [r["mb_coord_per_zn"] for r in rows if r["mb_coord_per_zn"]]
        angle_vals = [r["mb_mean_szns_angle"] for r in rows
                      if not math.isnan(r["mb_mean_szns_angle"])]
        zns_vals   = [r["mb_mean_zn_s_dist"] for r in rows
                      if not math.isnan(r["mb_mean_zn_s_dist"])]
        print(f"\nSummary statistics:")
        print(f"  Cys per Zn:     {statistics.mean(coord_vals):.2f} ± {statistics.stdev(coord_vals):.2f}  (ideal ~3.0)")
        if zns_vals:
            print(f"  Zn-S distance:  {statistics.mean(zns_vals):.3f} Å  (crystal avg ~2.33 Å)")
        if angle_vals:
            print(f"  ∠S-Zn-S:       {statistics.mean(angle_vals):.1f}°  (ideal tetrahedral 109.5°)")


if __name__ == "__main__":
    main()
