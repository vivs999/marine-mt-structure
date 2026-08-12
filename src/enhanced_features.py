"""Enhanced feature engineering for metallothionein classification.

Generates biology-informed features from sequence and structural data,
designed to improve habitat classification transfer performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import re


def compute_cysteine_spacing(sequence: str) -> Dict[str, float]:
    """Compute statistics on gaps between cysteine residues.
    
    Captures cysteine clustering patterns that may differ by habitat.
    """
    if not sequence:
        return {
            'cys_spacing_mean': 0.0,
            'cys_spacing_std': 0.0,
            'cys_spacing_min': 0.0,
            'cys_spacing_max': 0.0,
            'cys_spacing_cv': 0.0,
        }
    
    seq = sequence.upper()
    cys_positions = [i for i, aa in enumerate(seq) if aa == 'C']
    
    if len(cys_positions) < 2:
        return {
            'cys_spacing_mean': 0.0,
            'cys_spacing_std': 0.0,
            'cys_spacing_min': 0.0,
            'cys_spacing_max': 0.0,
            'cys_spacing_cv': 0.0,  # coefficient of variation
        }
    
    gaps = np.diff(cys_positions)
    mean_gap = float(np.mean(gaps))
    std_gap = float(np.std(gaps))
    cv = std_gap / mean_gap if mean_gap > 0 else 0.0
    
    return {
        'cys_spacing_mean': mean_gap,
        'cys_spacing_std': std_gap,
        'cys_spacing_min': float(np.min(gaps)),
        'cys_spacing_max': float(np.max(gaps)),
        'cys_spacing_cv': cv,
    }


def compute_cysteine_cluster_features(sequence: str) -> Dict[str, float]:
    """Identify cysteine clustering and positional bias.
    
    Features:
    - Maximum consecutive cysteines (CC, CCC patterns)
    - Cysteine position bias (N-terminal, C-terminal, center)
    - Cluster count (regions with >50% cysteine)
    """
    if not sequence:
        return {
            'max_consecutive_cys': 0,
            'cys_n_term_bias': 0.0,
            'cys_c_term_bias': 0.0,
            'cys_center_fraction': 0.0,
        }
    
    seq = sequence.upper()
    length = len(seq)
    
    # Max consecutive cysteines
    max_consecutive = 0
    current_consecutive = 0
    for aa in seq:
        if aa == 'C':
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    # Cysteine positions
    cys_positions = [i for i, aa in enumerate(seq) if aa == 'C']
    
    if not cys_positions:
        return {
            'max_consecutive_cys': 0,
            'cys_n_term_bias': 0.0,
            'cys_c_term_bias': 0.0,
            'cys_center_fraction': 0.0,
        }
    
    # Positional bias: fraction in first/last third
    n_term_count = sum(1 for p in cys_positions if p < length / 3)
    c_term_count = sum(1 for p in cys_positions if p > 2 * length / 3)
    center_count = len(cys_positions) - n_term_count - c_term_count
    
    total_cys = len(cys_positions)
    
    return {
        'max_consecutive_cys': max_consecutive,
        'cys_n_term_bias': n_term_count / total_cys if total_cys > 0 else 0.0,
        'cys_c_term_bias': c_term_count / total_cys if total_cys > 0 else 0.0,
        'cys_center_fraction': center_count / total_cys if total_cys > 0 else 0.0,
    }


def compute_local_context_features(sequence: str, window: int = 3) -> Dict[str, float]:
    """Compute amino acid composition in windows around cysteines.
    
    Captures local environment around metal-binding sites.
    """
    if not sequence:
        return {
            'cys_context_hydrophobic': 0.0,
            'cys_context_charged': 0.0,
            'cys_context_polar': 0.0,
            'cys_context_aromatic': 0.0,
        }
    
    seq = sequence.upper()
    cys_positions = [i for i, aa in enumerate(seq) if aa == 'C']
    
    if not cys_positions:
        return {
            'cys_context_hydrophobic': 0.0,
            'cys_context_charged': 0.0,
            'cys_context_polar': 0.0,
            'cys_context_aromatic': 0.0,
        }
    
    hydrophobic = set('AILMFVPW')
    charged = set('KRDE')
    polar = set('STNQY')
    aromatic = set('FWY')
    
    total_window_residues = 0
    hydrophobic_count = 0
    charged_count = 0
    polar_count = 0
    aromatic_count = 0
    
    for pos in cys_positions:
        start = max(0, pos - window)
        end = min(len(seq), pos + window + 1)
        
        for i in range(start, end):
            if i == pos:
                continue  # Skip the cysteine itself
            aa = seq[i]
            total_window_residues += 1
            if aa in hydrophobic:
                hydrophobic_count += 1
            if aa in charged:
                charged_count += 1
            if aa in polar:
                polar_count += 1
            if aa in aromatic:
                aromatic_count += 1
    
    if total_window_residues == 0:
        return {
            'cys_context_hydrophobic': 0.0,
            'cys_context_charged': 0.0,
            'cys_context_polar': 0.0,
            'cys_context_aromatic': 0.0,
        }
    
    return {
        'cys_context_hydrophobic': hydrophobic_count / total_window_residues,
        'cys_context_charged': charged_count / total_window_residues,
        'cys_context_polar': polar_count / total_window_residues,
        'cys_context_aromatic': aromatic_count / total_window_residues,
    }


def compute_charge_and_composition_features(sequence: str) -> Dict[str, float]:
    """Compute global charge, hydrophobicity, and composition features."""
    if not sequence:
        return {
            'positive_fraction': 0.0,
            'negative_fraction': 0.0,
            'charged_fraction': 0.0,
            'hydrophobic_fraction': 0.0,
            'polar_fraction': 0.0,
            'aromatic_fraction': 0.0,
            'small_fraction': 0.0,
            'proline_fraction': 0.0,
            'glycine_fraction': 0.0,
            'lysine_fraction': 0.0,
            'arginine_fraction': 0.0,
            'serine_fraction': 0.0,
            'net_charge_density': 0.0,
            'charge_asymmetry': 0.0,
        }
    
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        return {}
    
    # Charged residues
    positive = 'KR'
    negative = 'DE'
    positive_count = sum(seq.count(aa) for aa in positive)
    negative_count = sum(seq.count(aa) for aa in negative)
    charged_count = positive_count + negative_count
    
    # Hydrophobic residues
    hydrophobic = 'AILMFVPW'
    hydrophobic_count = sum(seq.count(aa) for aa in hydrophobic)
    
    # Polar residues
    polar = 'STNQ'
    polar_count = sum(seq.count(aa) for aa in polar)
    
    # Aromatic
    aromatic = 'FWY'
    aromatic_count = sum(seq.count(aa) for aa in aromatic)
    
    # Small residues
    small = 'AGSP'
    small_count = sum(seq.count(aa) for aa in small)
    
    # Proline (helix breaker)
    proline_count = seq.count('P')
    
    # Glycine (flexible)
    glycine_count = seq.count('G')
    
    # Lysine and arginine separately
    lysine_count = seq.count('K')
    arginine_count = seq.count('R')
    
    # Serine (common near metal sites)
    serine_count = seq.count('S')
    
    # Net charge
    net_charge = positive_count - negative_count
    
    return {
        'positive_fraction': positive_count / length,
        'negative_fraction': negative_count / length,
        'charged_fraction': charged_count / length,
        'hydrophobic_fraction': hydrophobic_count / length,
        'polar_fraction': polar_count / length,
        'aromatic_fraction': aromatic_count / length,
        'small_fraction': small_count / length,
        'proline_fraction': proline_count / length,
        'glycine_fraction': glycine_count / length,
        'lysine_fraction': lysine_count / length,
        'arginine_fraction': arginine_count / length,
        'serine_fraction': serine_count / length,
        'net_charge_density': net_charge / length,
        'charge_asymmetry': abs(positive_count - negative_count) / max(positive_count + negative_count, 1),
    }


def compute_sequence_motif_features(sequence: str) -> Dict[str, float]:
    """Count biologically relevant motifs in MT sequences.
    
    Common MT motifs:
    - CXC: common disulfide pattern
    - CXXC: wider spacing
    - KC, SC: charged/polar near cysteine
    - CC: adjacent cysteines
    """
    if not sequence:
        return {
            'cxc_motif_density': 0.0,
            'cxxc_motif_density': 0.0,
            'cc_motif_density': 0.0,
            'kc_motif_density': 0.0,
            'sc_motif_density': 0.0,
            'pc_motif_density': 0.0,
        }
    
    seq = sequence.upper()
    length = len(seq)
    
    if length == 0:
        return {}
    
    # Count motifs
    cxc_count = len(re.findall(r'C.C', seq))
    cxxc_count = len(re.findall(r'C..C', seq))
    cc_count = len(re.findall(r'CC', seq))
    kc_count = len(re.findall(r'KC', seq)) + len(re.findall(r'CK', seq))
    sc_count = len(re.findall(r'SC', seq)) + len(re.findall(r'CS', seq))
    pc_count = len(re.findall(r'PC', seq)) + len(re.findall(r'CP', seq))
    
    return {
        'cxc_motif_density': cxc_count / length,
        'cxxc_motif_density': cxxc_count / length,
        'cc_motif_density': cc_count / length,
        'kc_motif_density': kc_count / length,
        'sc_motif_density': sc_count / length,
        'pc_motif_density': pc_count / length,
    }


def compute_structural_nonlinear_features(
    n_cys_sulfur_atoms: float,
    mean_ss_distance: float,
    rg_cysteine_sulfur: float,
    cysteine_density: float,
) -> Dict[str, float]:
    """Compute additional nonlinear structural transforms.
    
    Beyond the 4 existing derived features, add more robust transforms
    that may capture habitat-specific structural compactness patterns.
    """
    features = {}
    
    # Squared and log transforms for scale invariance
    if mean_ss_distance > 0:
        features['log_mean_ss_distance'] = np.log1p(mean_ss_distance)
        features['mean_ss_distance_sq'] = mean_ss_distance ** 2
    else:
        features['log_mean_ss_distance'] = 0.0
        features['mean_ss_distance_sq'] = 0.0
    
    if rg_cysteine_sulfur > 0:
        features['log_rg_cysteine_sulfur'] = np.log1p(rg_cysteine_sulfur)
        features['rg_cysteine_sulfur_sq'] = rg_cysteine_sulfur ** 2
    else:
        features['log_rg_cysteine_sulfur'] = 0.0
        features['rg_cysteine_sulfur_sq'] = 0.0
    
    # Interaction terms
    if rg_cysteine_sulfur > 0 and mean_ss_distance > 0:
        features['ss_rg_ratio_sq'] = (mean_ss_distance / rg_cysteine_sulfur) ** 2
        features['compactness_index'] = n_cys_sulfur_atoms / (rg_cysteine_sulfur * mean_ss_distance)
    else:
        features['ss_rg_ratio_sq'] = 0.0
        features['compactness_index'] = 0.0
    
    # Normalized spread
    if n_cys_sulfur_atoms > 0:
        features['normalized_rg'] = rg_cysteine_sulfur / np.sqrt(n_cys_sulfur_atoms)
        features['normalized_ss'] = mean_ss_distance / np.sqrt(n_cys_sulfur_atoms)
    else:
        features['normalized_rg'] = 0.0
        features['normalized_ss'] = 0.0
    
    # Density-weighted metrics
    features['density_weighted_ss'] = cysteine_density * mean_ss_distance
    features['density_weighted_rg'] = cysteine_density * rg_cysteine_sulfur
    
    return features


def extract_enhanced_features(
    sequence: str,
    n_cys_sulfur_atoms: Optional[float] = None,
    mean_ss_distance: Optional[float] = None,
    rg_cysteine_sulfur: Optional[float] = None,
    cysteine_density: Optional[float] = None,
) -> Dict[str, float]:
    """Main entry point: extract all enhanced features from sequence + structure.
    
    Args:
        sequence: Amino acid sequence
        n_cys_sulfur_atoms: Number of cysteine sulfur atoms from structure
        mean_ss_distance: Mean pairwise S-S distance from structure
        rg_cysteine_sulfur: Radius of gyration of cysteine sulfurs
        cysteine_density: Cysteine fraction in sequence
    
    Returns:
        Dictionary of enhanced features
    """
    sequence = sequence or ""
    features = {}

    # Sequence-based features
    features.update(compute_cysteine_spacing(sequence))
    features.update(compute_cysteine_cluster_features(sequence))
    features.update(compute_local_context_features(sequence))
    features.update(compute_charge_and_composition_features(sequence))
    features.update(compute_sequence_motif_features(sequence))

    # Structural nonlinear transforms
    if all(x is not None for x in [n_cys_sulfur_atoms, mean_ss_distance, rg_cysteine_sulfur, cysteine_density]):
        structural_features = compute_structural_nonlinear_features(
            n_cys_sulfur_atoms,
            mean_ss_distance,
            rg_cysteine_sulfur,
            cysteine_density,
        )
    else:
        structural_features = compute_structural_nonlinear_features(0.0, 0.0, 0.0, 0.0)
    features.update(structural_features)
    
    return features


if __name__ == "__main__":
    # Test on a sample sequence
    test_seq = "MDPCECSKTGSCNCGGSCKCSNCACTSCKKSCCPCCPSDCSKCASGCVCKGKTCDTSCCQ"
    feats = extract_enhanced_features(
        test_seq,
        n_cys_sulfur_atoms=20.0,
        mean_ss_distance=11.9,
        rg_cysteine_sulfur=9.4,
        cysteine_density=0.33,
    )
    
    print("Enhanced features extracted:")
    for k, v in sorted(feats.items()):
        print(f"  {k:30s}: {v:.4f}")
    print(f"\nTotal features: {len(feats)}")
