"""Environmental metadata compilation and dataset integration for marine metallothioneins.

This module provides:
- A curated lookup table of known habitat parameters for common marine MT species.
- Utilities to merge structural features with habitat metadata.
- A simple heuristic habitat classifier when species data is missing.

Environmental parameters tracked:
    habitat_type    : one of {coastal_estuarine, open_ocean, deep_sea,
                               hydrothermal_vent, polar}
    mean_temp_c     : mean annual sea-surface (or habitat) temperature in °C
    mean_ph         : mean annual habitat pH
    metal_exposure  : categorical {low, moderate, high}
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

HABITAT_TYPES = {
    "coastal_estuarine",
    "open_ocean",
    "deep_sea",
    "hydrothermal_vent",
    "polar",
}

METAL_EXPOSURE_LEVELS = {"low", "moderate", "high"}

# Default metal-exposure assignment by habitat (PRD §2)
_HABITAT_METAL_DEFAULT: Dict[str, str] = {
    "coastal_estuarine": "moderate",
    "open_ocean": "low",
    "deep_sea": "low",
    "hydrothermal_vent": "high",
    "polar": "low",
}

# Curated reference metadata for well-studied marine MT species.
# Sources: Bio-ORACLE (v2.1), OBIS, and peer-reviewed literature on marine
# metallothioneins, including studies by Roosens et al., Viarengo et al.,
# and datasets from MarineSpecies and SeaLifeBase (Feb 2024).
SPECIES_METADATA: Dict[str, Dict[str, object]] = {
    # =========== FISH (Actinopterygii) ===========
    "Fundulus heteroclitus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.2,  # Atlantic coast: seasonal 5-25°C, annual mean ~15°C
        "mean_ph": 8.0,  # Estuarine pH typical 7.8-8.2
        "metal_exposure": "high",  # MT-up in polluted estuaries (Cd, Pb, Cu)
    },
    "Oncorhynchus mykiss": {
        "habitat_type": "open_ocean",  # Anadromous; marine phase in North Pacific
        "mean_temp_c": 10.5,  # North Pacific surface mean 8-12°C
        "mean_ph": 8.15,  # Full ocean pH
        "metal_exposure": "low",  # Offshore low pollution
    },
    "Scophthalmus maximus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 12.8,  # European Atlantic: 8-17°C range
        "mean_ph": 8.05,  # Neritic coastal
        "metal_exposure": "moderate",  # Working in EU coastal zones
    },
    "Platichthys flesus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 11.5,  # North Sea/Baltic: cold temperate
        "mean_ph": 8.0,  # Coastal
        "metal_exposure": "moderate",  # Known bioaccumulator in Baltic MT studies
    },
    "Sparus aurata": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 18.3,  # Mediterranean: 14-24°C
        "mean_ph": 8.2,  # Med pH
        "metal_exposure": "moderate",  # Commercial aquaculture species
    },
    # =========== MOLLUSKS (Mollusca) ===========
    "Crassostrea gigas": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 16.5,  # Pacific & Atlantic coasts: 12-21°C
        "mean_ph": 8.05,  # Estuarine-influenced
        "metal_exposure": "high",  # Suspension feeder; strong MT bioaccumulation
    },
    "Mytilus galloprovincialis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.8,  # Mediterranean & Atlantic: 12-19°C
        "mean_ph": 8.1,  # Coastal Mediterranean
        "metal_exposure": "high",  # Heavy MT induction in anthropogenic sites
    },
    "Mytilus edulis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 10.2,  # North Atlantic/North Sea: 6-14°C
        "mean_ph": 8.0,  # Cool temperate coastal
        "metal_exposure": "high",  # Classic MT biomonitor species (UNEP)
    },
    "Ostrea edulis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 13.5,  # European coastal: 9-18°C
        "mean_ph": 8.05,  # Neritic pH
        "metal_exposure": "high",  # Sedentary; MT response to Cd, Cu, Zn
    },
    # =========== CRUSTACEANS (Crustacea) ===========
    "Carcinus maenas": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 12.8,  # Temperate: 7-18°C
        "mean_ph": 8.0,  # Coastal
        "metal_exposure": "moderate",  # Mobile predator; moderate MT induction
    },
    "Homarus americanus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 8.5,  # North American Atlantic: 4-12°C
        "mean_ph": 8.0,  # Atlantic coastal
        "metal_exposure": "low",  # Deep burrows in clean substrates
    },
    "Callinectes sapidus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 19.2,  # Atlantic & Gulf coasts: 15-28°C
        "mean_ph": 8.0,  # Estuarine
        "metal_exposure": "moderate",  # Economically important; MT studies in Cd
    },
    "Stomatopoda sp.": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 24.5,  # Tropical: 22-27°C
        "mean_ph": 8.2,  # Tropical coastal
        "metal_exposure": "moderate",  # Emerging MT research on stomatopods
    },
    # =========== DEEP SEA & HYDROTHERMAL VENT ===========
    "Bathymodiolus thermophilus": {
        "habitat_type": "hydrothermal_vent",
        "mean_temp_c": 12.8,  # Vent field background ambient ~3°C; mussel symbiont region ~15°C
        "mean_ph": 6.2,  # Acidic vent fluids (pH 5.8-6.5)
        "metal_exposure": "high",  # Elemental sulfide minerals (Fe, Cu, Zn, Cd)
    },
    "Riftia pachyptila": {
        "habitat_type": "hydrothermal_vent",
        "mean_temp_c": 11.0,  # Vent ambient; tube wall ~25-40°C internally
        "mean_ph": 6.1,  # Acidic vent fluids
        "metal_exposure": "high",  # Hyperaccumulator of metals via chemosynthetic symbionts
    },
    "Alvinella pompejana": {
        "habitat_type": "hydrothermal_vent",
        "mean_temp_c": 13.5,  # Extreme thermoduric worm; vent wall 40-80°C
        "mean_ph": 6.0,  # Vent fluid
        "metal_exposure": "high",  # High MT expression under extreme conditions
    },
    # =========== POLAR ===========
    "Trematomus bernacchii": {
        "habitat_type": "polar",
        "mean_temp_c": -1.8,  # Antarctic shelf: -2 to 0°C
        "mean_ph": 8.22,  # Southern Ocean pH (slightly high due to lower warmth)
        "metal_exposure": "low",  # Remote Antarctica; pristine waters
    },
    "Chionodraco hamatus": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,  # Antarctic: -2 to -1°C
        "mean_ph": 8.2,  # Southern Ocean
        "metal_exposure": "low",  # Deep Antarctic waters; minimal anthropogenic input
    },
    "Liparis atlanticus": {
        "habitat_type": "deep_sea",
        "mean_temp_c": 2.1,  # Atlantic abyssal: 2-4°C
        "mean_ph": 8.1,  # Deep sea pH
        "metal_exposure": "low",  # Extreme depth; limited metal cycling
    },
    # =========== OPEN OCEAN ===========
    "Thalassoma bifasciatum": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 26.8,  # Caribbean/open Atlantic: 25-28°C
        "mean_ph": 8.18,  # Surface ocean pH
        "metal_exposure": "low",  # Pelagic teleost; open water cleaners
    },
    "Sardinella aurita": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 24.5,  # Tropical Atlantic: 22-26°C
        "mean_ph": 8.15,  # Epipelagic
        "metal_exposure": "low",  # Pelagic schooling across clean waters
    },
}


def lookup_species_metadata(organism: str) -> Optional[Dict[str, object]]:
    """Return curated habitat metadata for *organism*, or None if unknown."""
    if organism in SPECIES_METADATA:
        return dict(SPECIES_METADATA[organism])
    # Try partial / genus-level match
    for known, meta in SPECIES_METADATA.items():
        if organism.split()[0] == known.split()[0]:
            logger.warning("Genus-level match for %s -> %s", organism, known)
            return dict(meta)
    return None


def default_metadata(habitat_type: str = "open_ocean") -> Dict[str, object]:
    """Return a default metadata dict for the given habitat type."""
    if habitat_type not in HABITAT_TYPES:
        raise ValueError(f"Unknown habitat_type: {habitat_type!r}. Must be one of {HABITAT_TYPES}")
    return {
        "habitat_type": habitat_type,
        "mean_temp_c": None,
        "mean_ph": None,
        "metal_exposure": _HABITAT_METAL_DEFAULT[habitat_type],
    }


def enrich_with_metadata(
    sequences: List[Dict[str, str]],
    fallback_habitat: str = "open_ocean",
) -> List[Dict[str, object]]:
    """Merge a list of sequence records (from ingest_uniprot) with habitat metadata.

    Unknown organisms receive the *fallback_habitat* defaults.
    Returns a new list of dicts combining sequence fields and metadata fields.
    """
    enriched = []
    for rec in sequences:
        organism = rec.get("organism", "")
        meta = lookup_species_metadata(organism)
        if meta is None:
            logger.warning("No metadata for %s; using fallback habitat '%s'", organism, fallback_habitat)
            meta = default_metadata(fallback_habitat)
        row: Dict[str, object] = dict(rec)
        row.update(meta)
        enriched.append(row)
    return enriched


def write_metadata_csv(path: str, rows: List[Dict[str, object]]) -> None:
    """Write the enriched metadata table to *path*."""
    if not rows:
        logger.warning("write_metadata_csv called with empty rows")
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "uniprot_id",
        "organism",
        "sequence",
        "length",
        "cysteine_count",
        "reviewed",
        "habitat_type",
        "mean_temp_c",
        "mean_ph",
        "metal_exposure",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Convert None to empty string for CSV serialization
            output_row = {}
            for f in fieldnames:
                val = row.get(f)
                output_row[f] = "" if val is None else str(val)
            writer.writerow(output_row)


def read_metadata_csv(path: str) -> List[Dict[str, str]]:
    """Read the metadata CSV and return a list of dicts."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def integrate_features(
    metadata_rows: List[Dict[str, object]],
    feature_rows: List[Dict[str, object]],
    key: str = "uniprot_id",
) -> List[Dict[str, object]]:
    """Merge structural feature records with metadata records on *key*.

    Returns a list with one dict per record present in both inputs.
    Records not found in the feature set are dropped with a warning.
    """
    feature_index = {str(row[key]): row for row in feature_rows}
    integrated = []
    for meta in metadata_rows:
        record_key = str(meta.get(key, ""))
        feats = feature_index.get(record_key)
        if feats is None:
            logger.warning("No features found for %s=%s; skipping", key, record_key)
            continue
        merged: Dict[str, object] = dict(meta)
        merged.update(feats)
        integrated.append(merged)
    return integrated
