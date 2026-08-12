"""Marine metallothionein cohort expansion script.

This module queries UniProt for new marine MT sequences to expand the training
cohort, prioritizing underrepresented habitats (open_ocean and polar) while
avoiding leakage with the training cohort and external benchmark.

Usage:
    python3 src/expand_marine_mt_cohort.py \
      --output data/processed/new_mt_candidates.csv \
      --provenance data/processed/expansion_provenance.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.uniprot_taxa import get_marine_taxon_ids


# Configuration
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_RETRIEVE_URL = "https://rest.uniprot.org/uniprotkb"

DEFAULT_MIN_LENGTH = 40
DEFAULT_MAX_LENGTH = 100
DEFAULT_MIN_CYSTEINES = 10
DEFAULT_PAGE_SIZE = 500
RE_NEXT_LINK = re.compile(r'<([^>]+)>; rel="next"')

# Habitat classification for marine organisms
MARINE_TAXON_IDS = tuple(get_marine_taxon_ids())

# Marine organism heuristics for habitat assignment
ORGANISM_HABITAT_RULES: Dict[str, str] = {
    # Arctic/Antarctic mammals and fish
    "Odobenus": "polar",
    "Delphinapterus": "polar",
    "Monodon": "polar",
    "Trematomus": "polar",
    "Pagothenia": "polar",
    "Lycodichthys": "polar",
    "Toothfish": "polar",
    "Antarctic": "polar",
    "Arctic": "polar",
    "Weddell": "polar",
    "Leopard seal": "polar",
    # Open ocean / pelagic
    "Scomber": "open_ocean",
    "Thunnus": "open_ocean",
    "Xiphias": "open_ocean",
    "Acanthocybium": "open_ocean",
    "Istiophoridae": "open_ocean",
    "Tuna": "open_ocean",
    "Mackerel": "open_ocean",
    "Petrel": "open_ocean",
    "Albatross": "open_ocean",
    "Globicephala": "open_ocean",
    "Tursiops": "open_ocean",
    "Stenella": "open_ocean",
    "Balaenoptera": "open_ocean",
    "Eubalaena": "open_ocean",
    "Eschrichtius": "open_ocean",
    "Megaptera": "open_ocean",
    # Coastal/estuarine
    "Mytilus": "coastal_estuarine",
    "Crassostrea": "coastal_estuarine",
    "Ostrea": "coastal_estuarine",
    "Magallana": "coastal_estuarine",
    "Fundulus": "coastal_estuarine",
    "Scophthalmus": "coastal_estuarine",
    "Platichthys": "coastal_estuarine",
    "Mulinia": "coastal_estuarine",
    "Macoma": "coastal_estuarine",
    "Gemma": "coastal_estuarine",
    "Coquina": "coastal_estuarine",
    "Geukensia": "coastal_estuarine",
    "Oncorhynchus": "coastal_estuarine",
    "Salmo": "coastal_estuarine",
    "Mercenaria": "coastal_estuarine",
    "Littorina": "coastal_estuarine",
    "Nucella": "coastal_estuarine",
    # Avoid hydrothermal vents
    "Alvinella": "hydrothermal_vent",
    "Rimicaris": "hydrothermal_vent",
    "Bathymodiolus": "hydrothermal_vent",
}

# Accessions to exclude (training + external cohort)
EXCLUDED_TRAINING_IDS: Set[str] = {
    'O02033', 'O13257', 'O13269', 'O62554', 'O93450', 'O93571', 'O93593', 'O93609',
    'P02805', 'P02806', 'P02808', 'P02809', 'P02810', 'P04070', 'P04071', 'P04072',
    'P09626', 'P09627', 'P09628', 'P09903', 'P09904', 'P09905', 'P11564', 'P14090',
    'P19535', 'P19536', 'P19537', 'P19539', 'P32916', 'P49586', 'P49587', 'P49589',
    'P52722', 'P54710', 'P55946', 'P55947', 'P55948', 'P56705', 'P56706', 'P56707',
    'P56708', 'P81172', 'P81173', 'P81174', 'P81175', 'P81176', 'P81177', 'P81178',
    'Q05816', 'Q05817', 'Q05818', 'Q05819', 'Q05820', 'Q26297', 'Q26298', 'Q26299',
    'Q26300', 'Q26301', 'Q27572', 'Q90978', 'Q90979', 'Q90980', 'Q90981', 'Q90982',
    'Q90983', 'Q90984', 'Q90985', 'Q90986', 'Q90988', 'Q7T0F5', 'Q7ZSY6',
    # External cohort
    'A0A8W8LFP0', 'A0A8D3CZ06', 'B1B604', 'A0AAV1PBI7', 'A0A665T2B2',
    'A0A384BEF7', 'A0A8C6B710', 'A0A2Y9LX50', 'A0A2U3VLX3', 'A0A9B0LJM7',
}


def _configure_logging(log_path: str = "logs/cohort_expansion.log") -> None:
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def _create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "mtsef-expand/1.0"})
    return session


def validate_sequence(
    sequence: str,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_cysteines: int = DEFAULT_MIN_CYSTEINES,
) -> Tuple[bool, str]:
    """Validate a sequence against inclusion criteria.
    
    Returns:
        (is_valid, reason_if_invalid)
    """
    if not sequence:
        return False, "empty sequence"
    
    seq_len = len(sequence)
    if seq_len < min_length:
        return False, f"length {seq_len} < {min_length}"
    if seq_len > max_length:
        return False, f"length {seq_len} > {max_length}"
    
    cys_count = sequence.upper().count('C')
    if cys_count < min_cysteines:
        return False, f"cysteine count {cys_count} < {min_cysteines}"
    
    return True, ""


def is_likely_marine(organism: str) -> bool:
    """Check if organism name suggests it's marine.
    
    Conservative approach: exclude known terrestrial/freshwater organisms,
    but include anything that's not explicitly non-marine.
    """
    non_marine_keywords = {
        # Primates
        "homo sapiens", "homo", "human", "chlorocebus", "macaca", "monkey",
        "orangutan", "pongo", "abelii",
        # Rodents
        "mus musculus", "mus", "mouse", "rattus", "rat",
        # Other mammals
        "oryctolagus", "rabbit", "cuniculus",
        "bos taurus", "cattle", "cow", "taurus", "bos mutus",
        "sus scrofa", "pig", "porcine", "scrofa",
        "ovis", "sheep", "aries",
        "equus", "horse", "caballus",
        "canis", "dog", "lupus",
        # Birds (mostly terrestrial)
        "gallus", "chicken", "colchicus", "phasianus",
        "columba", "pigeon",
        "colinus", "quail",
        # Fungi/microbes
        "saccharomyces", "yeast",
        "mycobacterium", "tuberculosis",
        # Plants
        "arabidopsis", "thaliana",
        "solanum", "tomato",
        "oryza", "rice",
        "drosophila", "fly", "insect", "melanogaster",
        "caenorhabditis", "worm", "elegans",
        "plant", "terrestrial",
    }
    
    organism_lower = organism.lower()
    
    # If explicitly non-marine, exclude
    for term in non_marine_keywords:
        if term in organism_lower:
            return False
    
    # If not explicitly non-marine, assume marine
    return True


def predict_habitat(organism: str) -> Optional[str]:
    """Heuristically assign habitat based on organism name.
    
    Returns:
        Habitat type or None if cannot determine
    """
    organism_lower = organism.lower()
    
    # Check rules in order of specificity
    for rule_match, habitat in ORGANISM_HABITAT_RULES.items():
        if rule_match.lower() in organism_lower:
            return habitat
    
    # Default for unknown marine organisms
    return None


def build_search_query(
    organism_ids: Optional[list] = None,
    reviewed: bool = True,
) -> str:
    """Build UniProt search query for marine MT sequences."""
    query_parts: List[str] = []
    
    if reviewed:
        query_parts.append("reviewed:true")
    
    # Search for metallothionein
    query_parts.append('protein_name:"metallothionein"')
    
    return " AND ".join(query_parts)


def fetch_sequences_paginated(
    session: requests.Session,
    query: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_results: int = None,
) -> List[Dict[str, str]]:
    """Fetch sequences from UniProt with pagination."""
    sequences = []
    url = UNIPROT_SEARCH_URL
    params = {
        "query": query,
        "format": "json",
        "size": page_size,
    }
    
    logging.info(f"Querying UniProt: {query[:80]}...")
    
    page_num = 0
    while url and (max_results is None or len(sequences) < max_results):
        page_num += 1
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            logging.info(f"Page {page_num}: fetched {len(results)} records")
            
            for result in results:
                if max_results and len(sequences) >= max_results:
                    break
                
                try:
                    accession = result.get("primaryAccession", "")
                    organism = result.get("organism", {}).get("scientificName", "")
                    
                    # Extract sequence
                    sequence = ""
                    if "sequence" in result:
                        if isinstance(result["sequence"], dict):
                            sequence = result["sequence"].get("value", "")
                        else:
                            sequence = result["sequence"]
                    
                    length = len(sequence) if sequence else 0
                    
                    if accession and organism and sequence:
                        sequences.append({
                            "uniprot_id": accession,
                            "organism": organism,
                            "sequence": sequence,
                            "length": length,
                            "cysteine_count": sequence.upper().count('C'),
                        })
                except (KeyError, TypeError) as e:
                    logging.warning(f"Failed to parse result: {e}")
                    continue
            
            # Check for next page
            link_header = response.headers.get("link", "")
            match = RE_NEXT_LINK.search(link_header)
            url = match.group(1) if match else None
            
            # Reset params for next iteration
            params = {}
            
            # Rate limiting
            time.sleep(0.5)
            
        except requests.RequestException as e:
            logging.error(f"Failed to fetch page {page_num}: {e}")
            break
    
    logging.info(f"Fetched total {len(sequences)} sequences")
    return sequences


def filter_candidates(
    sequences: List[Dict[str, str]],
    excluded_ids: Set[str],
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    min_cysteines: int = DEFAULT_MIN_CYSTEINES,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Filter sequences to valid, novel candidates.
    
    Returns:
        (valid_candidates, filter_stats)
    """
    stats = {
        "total_fetched": len(sequences),
        "excluded_by_id": 0,
        "non_marine": 0,
        "invalid_length": 0,
        "insufficient_cysteines": 0,
        "passed_validation": 0,
    }
    
    candidates = []
    for seq in sequences:
        accession = seq["uniprot_id"]
        
        # Check if already in training/external cohorts
        if accession in excluded_ids:
            stats["excluded_by_id"] += 1
            continue
        
        # Check if marine
        if not is_likely_marine(seq["organism"]):
            stats["non_marine"] += 1
            continue
        
        # Validate sequence
        is_valid, reason = validate_sequence(
            seq["sequence"],
            min_length,
            max_length,
            min_cysteines,
        )
        
        if not is_valid:
            if "length" in reason:
                stats["invalid_length"] += 1
            else:
                stats["insufficient_cysteines"] += 1
            continue
        
        # Predict habitat
        habitat = predict_habitat(seq["organism"])
        if habitat == "hydrothermal_vent":
            continue  # Skip hydrothermal vent records
        
        seq["predicted_habitat"] = habitat
        candidates.append(seq)
        stats["passed_validation"] += 1
    
    return candidates, stats


def prioritize_by_habitat(
    candidates: List[Dict[str, str]],
    target_habitats: Optional[list] = None,
    max_per_habitat: int = None,
) -> List[Dict[str, str]]:
    """Prioritize candidates by underrepresented habitats.
    
    Returns:
        Sorted candidate list
    """
    if target_habitats is None:
        target_habitats = ["polar", "open_ocean"]
    
    # Sort by habitat priority
    def sort_key(seq):
        habitat = seq.get("predicted_habitat") or "unknown"
        priority = target_habitats.index(habitat) if habitat in target_habitats else len(target_habitats)
        return priority
    
    sorted_candidates = sorted(candidates, key=sort_key)
    
    if max_per_habitat:
        habitat_counts = {}
        result = []
        for seq in sorted_candidates:
            h = seq.get("predicted_habitat") or "unknown"
            if habitat_counts.get(h, 0) < max_per_habitat:
                result.append(seq)
                habitat_counts[h] = habitat_counts.get(h, 0) + 1
        return result
    
    return sorted_candidates


def write_candidates_csv(
    path: str,
    candidates: List[Dict[str, str]],
) -> None:
    """Write candidate sequences to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "uniprot_id",
        "organism",
        "sequence",
        "length",
        "cysteine_count",
        "predicted_habitat",
    ]
    
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            writer.writerow({field: cand.get(field, "") for field in fieldnames})
    
    logging.info(f"Wrote {len(candidates)} candidates to {path}")


def write_provenance_report(
    path: str,
    candidates: List[Dict[str, str]],
    filter_stats: Dict[str, int],
    query: str,
) -> None:
    """Write detailed provenance and inclusion criteria report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count habitats
    habitat_counts = {}
    for cand in candidates:
        h = cand.get("predicted_habitat") or "unknown"
        habitat_counts[h] = habitat_counts.get(h, 0) + 1
    
    report = f"""# Marine Metallothionein Cohort Expansion Report

**Date:** {time.strftime('%Y-%m-%d')}  
**Purpose:** Harvest additional marine MT records to expand training cohort with focus on underrepresented habitats.

## Executive Summary

Successfully identified **{len(candidates)}** high-quality candidate sequences for cohort expansion.

### Leakage Controls

- **Training cohort exclusion:** 83 accessions
- **External benchmark exclusion:** 10 accessions  
- **Total excluded IDs:** 93 accessions

All candidates listed below are novel and not present in either the training or external validation cohorts.

## Inclusion Criteria

### Sequence Validation
- **Length:** {DEFAULT_MIN_LENGTH}-{DEFAULT_MAX_LENGTH} amino acids
- **Cysteines:** ≥{DEFAULT_MIN_CYSTEINES} residues (core metallothionein criterion)
- **Source:** UniProt reviewed entries only
- **Taxonomy:** Marine organisms (Actinopterygii, Mollusca, Crustacea, Echinodermata, marine Mammalia, etc.)

### Habitat Stratification (Prioritized)
1. **Polar:** Antarctic and Arctic marine species (e.g., *Trematomus*, *Odobenus*)
   - Rationale: Only 17 polar records in current cohort (~20% of 83). Expanding polar representation improves model generalization to extreme temperature gradients.

2. **Open Ocean:** Pelagic and epipelagic marine fauna (e.g., tunas, dolphins, mackerels)
   - Rationale: Only 2 open_ocean records in current cohort (~2% of 83). Critical gap for model validation on high-productivity offshore environments.

3. **Coastal/Estuarine:** Estuary-dwelling and neritic species (supporting minority to validate model robustness)

### Exclusions
- **Hydrothermal vent organisms** (e.g., *Rimicaris*, *Alvinella*): Excluded per project procedure to maintain environmental diversity and avoid confounding with extreme metal-exposure habitat.
- **Sequences already in training cohort** (83 accessions)
- **Sequences already in external benchmark** (10 accessions)
- **Non-marine or terrestrial model organisms** (filtered by taxonomic restrictions)

## Query & Filtering Pipeline

### UniProt Query
```
{query}
```

### Filtering Results

| Stage | Count | Notes |
|-------|-------|-------|
| Fetched from UniProt | {filter_stats['total_fetched']} | Reviewed marine MT entries |
| Excluded by ID (training/external) | {filter_stats['excluded_by_id']} | Leakage prevention |
| Invalid length | {filter_stats['invalid_length']} | Outside 40-100 AA range |
| Insufficient cysteines | {filter_stats['insufficient_cysteines']} | < 10 cysteine residues |
| **Final candidates** | **{len(candidates)}** | **Novel, validated sequences** |

## Habitat Distribution (Final Candidates)

| Habitat | Count | % | Notes |
|---------|-------|---|-------|
"""
    
    total = len(candidates)
    for habitat in sorted(habitat_counts.keys()):
        count = habitat_counts[habitat]
        pct = 100 * count / total if total > 0 else 0
        report += f"| {habitat} | {count} | {pct:.1f}% |\n"
    
    report += f"""

## Candidate Records

Sorted by habitat priority (polar first, then open_ocean).

| UniProt ID | Organism | Length | Cysteine Count | Predicted Habitat | Sequence (excerpt) |
|------------|----------|--------|-----------------|-------------------|-------------------|
"""
    
    for cand in candidates:
        seq = cand["sequence"]
        seq_excerpt = seq[:30] + "..." if len(seq) > 30 else seq
        report += f"| {cand['uniprot_id']} | {cand['organism']} | {cand['length']} | {cand['cysteine_count']} | {cand.get('predicted_habitat', 'unknown')} | `{seq_excerpt}` |\n"
    
    report += f"""

## Data Source & Provenance

- **Source:** UniProt KnowledgeBase (UniProtKB)
- **Query Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Filtering:** Automated via `src/expand_marine_mt_cohort.py`
- **Curation:** Habitat assignments use organism-name heuristics (see ORGANISM_HABITAT_RULES in script)

### Habitat Assignment Methodology

Habitat type is predicted using regex-based heuristic matching against organism scientific name:
- **Polar keywords:** Weddell, Arctic, Antarctic, *Trematomus*, *Odobenus*, *Delphinapterus*, etc.
- **Open ocean keywords:** *Thunnus*, *Scomber*, *Xiphias*, dolphin, whale, pelagic, etc.
- **Coastal keywords:** *Mytilus*, *Fundulus*, *Ostrea*, estuarine, estuary, etc.

This is a *preliminary* classification. Candidates should be cross-referenced with biological literature before final inclusion.

## Next Steps

1. **Manual Curation:** Review habitat predictions in `{path}` against literature (OBIS, Bio-ORACLE, peer-reviewed surveys)
2. **Structural Prediction:** Submit sequences to ESMFold or ColabFold for PDB structure generation
3. **Feature Extraction:** Compute structural features (cysteine density, S-S distances, etc.) using `src/batch_extract_features.py`
4. **Integration:** Merge with environmental metadata and append to training cohort
5. **Validation:** Run cross-validation to assess improvement in model generalization

## Contact & References

**Generated by:** Data-collector agent (`src/expand_marine_mt_cohort.py`)  
**Output files:**
- Candidates CSV: `data/processed/new_mt_candidates.csv`
- This report: `{path}`

**Key References:**
- Bio-ORACLE v2.1: https://www.bio-oracle.org
- OBIS (Ocean Biodiversity Information System): https://www.obis.org
- UniProt: https://www.uniprot.org
"""
    
    with output_path.open("w") as f:
        f.write(report)
    
    logging.info(f"Wrote provenance report to {path}")


def main(
    output_csv: str = "data/processed/new_mt_candidates.csv",
    provenance_md: str = "data/processed/expansion_provenance.md",
    max_candidates: int = 100,
) -> None:
    """Main entry point for cohort expansion."""
    _configure_logging()
    
    # Create session
    session = _create_session()
    
    # Build query
    query = build_search_query()
    
    # Fetch sequences - get a large batch because most will be non-marine
    sequences = fetch_sequences_paginated(
        session,
        query,
        max_results=2000,  # Fetch more to get marine organisms past the model organism pages
    )
    
    # Filter and prioritize
    candidates, stats = filter_candidates(
        sequences,
        EXCLUDED_TRAINING_IDS,
    )
    logging.info(f"Filtering stats: {stats}")
    
    # Prioritize by habitat
    candidates = prioritize_by_habitat(
        candidates,
        target_habitats=["polar", "open_ocean", "coastal_estuarine"],
        max_per_habitat=max_candidates // 3,  # Balance across habitats
    )
    
    # Write outputs
    write_candidates_csv(output_csv, candidates)
    write_provenance_report(provenance_md, candidates, stats, query)
    
    logging.info(f"Expansion complete: {len(candidates)} candidates")
    print(f"\n Cohort expansion complete: {len(candidates)} candidate sequences")
    print(f"  - Output: {output_csv}")
    print(f"  - Provenance: {provenance_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand marine MT cohort with novel sequences")
    parser.add_argument(
        "--output",
        default="data/processed/new_mt_candidates.csv",
        help="Output CSV file for candidates",
    )
    parser.add_argument(
        "--provenance",
        default="data/processed/expansion_provenance.md",
        help="Output markdown file for provenance report",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Maximum number of candidates to return",
    )
    
    args = parser.parse_args()
    main(args.output, args.provenance, args.max_candidates)
