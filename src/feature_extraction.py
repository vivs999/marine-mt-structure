import math
from typing import List, Tuple, Optional

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XLE": "J",
}


def parse_sequence_from_pdb(pdb_text: str) -> str:
    """Extract the amino acid sequence from PDB ATOM records (CA atoms)."""
    seen: dict[int, str] = {}
    for line in pdb_text.splitlines():
        if len(line) < 26:
            continue
        record = line[0:6].strip()
        if record != "ATOM":
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        res_name = line[17:20].strip()
        try:
            res_seq = int(line[22:26].strip())
        except ValueError:
            continue
        if res_seq not in seen:
            seen[res_seq] = _THREE_TO_ONE.get(res_name, "X")
    if not seen:
        return ""
    return "".join(seen[k] for k in sorted(seen))


def cysteine_density(sequence: str) -> float:
    if not sequence:
        return 0.0
    count = sequence.count("C")
    return count / len(sequence)


def parse_pdb_sulfur_coords(pdb_text: str) -> List[Tuple[float, float, float]]:
    coords = []
    for line in pdb_text.splitlines():
        if len(line) < 54:
            continue
        record = line[0:6].strip()
        if record not in ("ATOM", "HETATM"):
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        if res_name == "CYS" and atom_name in ("SG", "S"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    return coords


def pairwise_distances(coords: List[Tuple[float, float, float]]) -> List[float]:
    dists = []
    n = len(coords)
    for i in range(n):
        xi, yi, zi = coords[i]
        for j in range(i + 1, n):
            xj, yj, zj = coords[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            dists.append(d)
    return dists


def mean_ss_distance(coords: List[Tuple[float, float, float]]) -> Optional[float]:
    dists = pairwise_distances(coords)
    if not dists:
        return None
    return sum(dists) / len(dists)


def radius_of_gyration(coords: List[Tuple[float, float, float]]) -> Optional[float]:
    if not coords:
        return None
    n = len(coords)
    cx = sum(x for x, _, _ in coords) / n
    cy = sum(y for _, y, _ in coords) / n
    cz = sum(z for _, _, z in coords) / n
    sqs = 0.0
    for x, y, z in coords:
        dx = x - cx
        dy = y - cy
        dz = z - cz
        sqs += dx * dx + dy * dy + dz * dz
    return math.sqrt(sqs / n)


def extract_features_from_pdb_and_sequence(pdb_text: str, sequence: Optional[str] = None) -> dict:
    """Extract structural and enhanced features from PDB and sequence.
    
    Uses the enhanced_features module for all sequence-derived and structural
    nonlinear features.
    
    Args:
        pdb_text: PDB file content
        sequence: Optional amino acid sequence
    
    Returns:
        Dictionary of features including:
        - n_cys_sulfur_atoms: Number of cysteine sulfur atoms
        - mean_ss_distance: Mean pairwise S-S distance
        - rg_cysteine_sulfur: Radius of gyration of cysteine sulfurs
        - cysteine_density: Fraction of cysteines in sequence (if sequence provided)
        - Plus all enhanced features from src.enhanced_features (if sequence provided)
    """
    coords = parse_pdb_sulfur_coords(pdb_text)
    n_cys = len(coords)
    mean_ss = mean_ss_distance(coords)
    rg = radius_of_gyration(coords)
    
    features = {
        "n_cys_sulfur_atoms": n_cys,
        "mean_ss_distance": mean_ss,
        "rg_cysteine_sulfur": rg,
    }
    
    if sequence is not None:
        cys_dens = cysteine_density(sequence)
        features["cysteine_density"] = cys_dens
        
        # Import and use the enhanced features
        from src.enhanced_features import extract_enhanced_features
        enhanced = extract_enhanced_features(
            sequence=sequence,
            n_cys_sulfur_atoms=n_cys,
            mean_ss_distance=mean_ss,
            rg_cysteine_sulfur=rg,
            cysteine_density=cys_dens,
        )
        features.update(enhanced)
    
    return features


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <pdb_file> [sequence]")
        sys.exit(1)
    pdb_path = sys.argv[1]
    seq = sys.argv[2] if len(sys.argv) >= 3 else None
    with open(pdb_path, "r") as f:
        pdb_text = f.read()
    feats = extract_features_from_pdb_and_sequence(pdb_text, seq)
    for k, v in feats.items():
        print(f"{k}: {v}")
