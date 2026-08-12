#!/usr/bin/env python3
"""
Enrich 250 metallothionein sequences with environmental metadata.

This script reads uniprot_sequences_marine.csv and assigns environmental parameters
to each organism using a multi-tier strategy:
  1. Curated lookup table (SPECIES_METADATA from metadata.py)
  2. Heuristic classification by organism type (fish, mollusk, crustacean, etc.)
  3. Default fallback values

Output: data/processed/metadata.csv with columns:
  - organism
  - habitat_type
  - temperature_celsius
  - ph
  - metal_exposure
  - data_source (e.g., "curated_lookup", "heuristic", "default")
  - notes (imputation notes if applicable)
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Assume metadata.py is in the same directory
from src.metadata import (
    SPECIES_METADATA,
    HABITAT_TYPES,
    METAL_EXPOSURE_LEVELS,
    lookup_species_metadata,
    default_metadata,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# TIER 2: HEURISTIC RULES FOR COMMON ORGANISM TYPES
# ============================================================================

HEURISTIC_RULES = {
    # Fish (Actinopterygii) - general marine fish
    "Danio rerio": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 27.0,  # Zebrafish native to SE Asia, tropical
        "mean_ph": 7.8,  # Freshwater/brackish tolerance
        "metal_exposure": "low",
    },
    "Gadus morhua": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 4.0,  # Atlantic cod, cold water
        "mean_ph": 8.0,
        "metal_exposure": "moderate",
    },
    "Dicentrarchus labrax": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.0,  # European sea bass
        "mean_ph": 8.0,
        "metal_exposure": "moderate",
    },
    "Ictalurus punctatus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 20.0,  # Channel catfish, North America
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Oreochromis mossambicus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 25.0,  # African tilapia, warm waters
        "mean_ph": 7.5,
        "metal_exposure": "moderate",
    },
    "Oreochromis aureus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 25.0,
        "mean_ph": 7.5,
        "metal_exposure": "moderate",
    },
    "Pagrus major": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 18.0,  # Red seabream, NW Pacific
        "mean_ph": 8.1,
        "metal_exposure": "moderate",
    },
    "Plecoglossus altivelis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.0,  # Ayu, East Asia
        "mean_ph": 7.8,
        "metal_exposure": "low",
    },
    "Pleuronectes platessa": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 8.0,  # European plaice, cold North Sea
        "mean_ph": 8.0,
        "metal_exposure": "moderate",
    },
    "Pseudopleuronectes americanus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 8.0,  # Winter flounder, Atlantic
        "mean_ph": 8.0,
        "metal_exposure": "moderate",
    },
    "Perca fluviatilis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 12.0,  # European perch, temperate freshwater
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Chelon auratus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 20.0,  # Golden grey mullet
        "mean_ph": 8.0,
        "metal_exposure": "moderate",
    },
    "Morone saxatilis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.0,  # Striped bass, Atlantic
        "mean_ph": 8.0,
        "metal_exposure": "moderate",
    },
    "Rutilus rutilus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 10.0,  # Common roach, temperate
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Esox lucius": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 10.0,  # Northern pike, temperate
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Scyliorhinus torazame": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.0,  # Japanese catshark
        "mean_ph": 8.0,
        "metal_exposure": "low",
    },
    "Zoarces viviparus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 6.0,  # Eelpout, Atlantic cold water
        "mean_ph": 8.0,
        "metal_exposure": "low",
    },
    "Xenopus laevis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 20.0,  # African clawed frog, freshwater
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Ambystoma mexicanum": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 16.0,  # Axolotl, freshwater
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    # Mollusks (Mollusca)
    "Mytilus galloprovincialis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.8,
        "mean_ph": 8.1,
        "metal_exposure": "high",
    },
    "Crassostrea virginica": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 16.0,  # Eastern oyster, Atlantic
        "mean_ph": 8.0,
        "metal_exposure": "high",
    },
    "Perna viridis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 28.0,  # Green mussel, tropical SE Asia
        "mean_ph": 8.1,
        "metal_exposure": "high",
    },
    "Dreissena polymorpha": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 12.0,  # Zebra mussel, temperate
        "mean_ph": 8.0,
        "metal_exposure": "high",
    },
    "Helix pomatia": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 12.0,  # Roman snail, temperate
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Arianta arbustorum": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 10.0,  # Alpine snail
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    # Crustaceans (Crustacea)
    "Scylla serrata": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 26.0,  # Mud crab, tropical SE Asia
        "mean_ph": 8.0,
        "metal_exposure": "moderate",
    },
    "Astacus astacus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 10.0,  # European crayfish, temperate
        "mean_ph": 7.8,
        "metal_exposure": "low",
    },
    "Potamon potamios": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.0,  # Greek freshwater crab
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    # Echinoderms
    "Strongylocentrotus purpuratus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 12.0,  # Purple sea urchin, California
        "mean_ph": 8.1,
        "metal_exposure": "moderate",
    },
    "Sphaerechinus granularis": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 14.0,  # Purple sea urchin, Atlantic
        "mean_ph": 8.1,
        "metal_exposure": "moderate",
    },
    "Paracentrotus lividus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 15.0,  # Edible sea urchin, Mediterranean
        "mean_ph": 8.1,
        "metal_exposure": "moderate",
    },
    "Lytechinus pictus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 14.0,  # Red sea urchin, Pacific
        "mean_ph": 8.1,
        "metal_exposure": "moderate",
    },
    "Sterechinus neumayeri": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,  # Antarctic sea urchin
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    # Antarctic Fish (Nototheniidae)
    "Trematomus bernacchii": {
        "habitat_type": "polar",
        "mean_temp_c": -1.8,
        "mean_ph": 8.22,
        "metal_exposure": "low",
    },
    "Chionodraco hamatus": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    "Chionodraco rastrospinosus": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    "Parachaenichthys charcoti": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    "Gymnodraco acuticeps": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    "Chaenocephalus aceratus": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    "Notothenia neglecta": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    "Pagothenia borchgrevinki": {
        "habitat_type": "polar",
        "mean_temp_c": -1.5,
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    # Marine mammals
    "Balaena mysticetus": {
        "habitat_type": "polar",
        "mean_temp_c": 0.0,  # Bowhead whale, Arctic
        "mean_ph": 8.2,
        "metal_exposure": "low",
    },
    "Stenella coeruleoalba": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 18.0,  # Striped dolphin, subtropical/temperate
        "mean_ph": 8.15,
        "metal_exposure": "low",
    },
    # Terrestrial / Freshwater (will get open_ocean default)
    "Homo sapiens": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Mus musculus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Rattus norvegicus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Sus scrofa": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Bos taurus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Canis lupus familiaris": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Oryctolagus cuniculus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Ovis aries": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Equus caballus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Pongo abelii": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Macaca fascicularis": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Chlorocebus aethiops": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Bos mutus grunniens": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Mesocricetus auratus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Cricetulus griseus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Cricetulus longicaudatus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    # Birds
    "Gallus gallus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Meleagris gallopavo": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Phasianus colchicus colchicus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Colinus virginianus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Columba livia": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Anas platyrhynchos": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Cairina moschata": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    # Insects
    "Drosophila melanogaster": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Drosophila simulans": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Drosophila ananassae": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Eisenia fetida": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Tetrahymena pyriformis": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Tetrahymena pigmentosa": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    # Caenorhabditis elegans
    "Caenorhabditis elegans": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    # Plants (will get open_ocean default)
    "Oryza sativa subsp. japonica": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Oryza sativa subsp. indica": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Arabidopsis thaliana": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Zea mays": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Triticum aestivum": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Solanum lycopersicum": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Vicia faba": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Pisum sativum": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Brassica juncea": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Brassica napus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Brassica campestris": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Brassica rapa subsp. pekinensis": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Trifolium repens": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Cicer arietinum": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Hordeum vulgare": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Malus domestica": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Prunus avium": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Fragaria ananassa": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Carica papaya": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Musa acuminata": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Coffea arabica": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Actinidia deliciosa": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Ricinus communis": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Nicotiana plumbaginifolia": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Nicotiana glutinosa": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Colocasia esculenta": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Casuarina glauca": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Picea glauca": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Erythranthe guttata": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Festuca rubra": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    # Reptiles
    "Podarcis siculus": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    # Fungi
    "Saccharomyces cerevisiae (strain ATCC 204508 / S288c)": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Schizosaccharomyces pombe (strain 972 / ATCC 24843)": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Candida glabrata (strain ATCC 2001 / BCRC 20586 / JCM 3761 / NBRC 0622 / NRRL Y-65 / CBS 138)": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Yarrowia lipolytica (strain CLIB 122 / E 150)": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    # Bacteria
    "Synechococcus sp": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Synechococcus elongatus (strain ATCC 33912 / PCC 7942 / FACHB-805)": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Pseudomonas fluorescens (strain Q2-87)": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Mycobacterium tuberculosis (strain CDC 1551 / Oshkosh)": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    "Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv)": {
        "habitat_type": "open_ocean",
        "mean_temp_c": 15.0,
        "mean_ph": 8.1,
        "metal_exposure": "low",
    },
    # Hydrothermal vent species (deep sea)
    "Thermostichus vulcanus": {
        "habitat_type": "hydrothermal_vent",
        "mean_temp_c": 12.0,
        "mean_ph": 6.5,
        "metal_exposure": "high",
    },
    "Thermarces cerberus": {
        "habitat_type": "hydrothermal_vent",
        "mean_temp_c": 12.0,
        "mean_ph": 6.5,
        "metal_exposure": "high",
    },
    "Cyprinus carpio": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 18.0,  # Common carp, temperate
        "mean_ph": 7.5,
        "metal_exposure": "moderate",
    },
    "Carassius auratus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 18.0,  # Goldfish
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Carassius cuvieri": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 18.0,
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Cyprinodon sp.": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 20.0,  # Killifish, warm
        "mean_ph": 7.5,
        "metal_exposure": "moderate",
    },
    "Barbatula barbatula": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 10.0,  # Stone loach, temperate
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
    "Gobiomorphus cotidianus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 12.0,  # New Zealand goby
        "mean_ph": 7.8,
        "metal_exposure": "low",
    },
    "Lithognathus mormyrus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 16.0,  # Striped seabream
        "mean_ph": 8.0,
        "metal_exposure": "moderate",
    },
    "Oryzias latipes": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 22.0,  # Japanese medaka
        "mean_ph": 7.8,
        "metal_exposure": "low",
    },
    "Salmo salar": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 12.0,  # Atlantic salmon
        "mean_ph": 8.0,
        "metal_exposure": "low",
    },
    "Salvelinus alpinus": {
        "habitat_type": "coastal_estuarine",
        "mean_temp_c": 8.0,  # Arctic char, cold
        "mean_ph": 7.5,
        "metal_exposure": "low",
    },
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_organism_metadata(organism: str) -> Tuple[Dict, str]:
    """
    Get metadata for an organism using a multi-tier strategy.

    Returns:
        (metadata_dict, data_source) where data_source is one of:
        - "curated_lookup" (from SPECIES_METADATA)
        - "heuristic" (from HEURISTIC_RULES)
        - "default" (fallback)
    """
    # Tier 1: Curated lookup
    meta = lookup_species_metadata(organism)
    if meta is not None:
        return meta, "curated_lookup"

    # Tier 2: Heuristic rules
    if organism in HEURISTIC_RULES:
        return dict(HEURISTIC_RULES[organism]), "heuristic"

    # Tier 3: Default fallback
    return default_metadata("open_ocean"), "default"


def enrich_sequences(input_csv: str, output_csv: str) -> Dict[str, object]:
    """
    Read sequences from input_csv, enrich with metadata, and write to output_csv.

    Returns a summary dict with stats.
    """
    sequences = []
    with open(input_csv, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sequences.append(row)

    logger.info(f"Loaded {len(sequences)} sequences from {input_csv}")

    # Enrich each sequence
    enriched = []
    organism_counts = {}
    habitat_counts = {}
    data_source_counts = {}

    for seq in sequences:
        organism = seq.get("organism", "")
        uniprot_id = seq.get("uniprot_id", "")

        # Get metadata
        metadata, source = get_organism_metadata(organism)

        # Build enriched row
        row = {
            "organism": organism,
            "habitat_type": metadata.get("habitat_type", "unknown"),
            "mean_temp_c": metadata.get("mean_temp_c", ""),
            "mean_ph": metadata.get("mean_ph", ""),
            "metal_exposure": metadata.get("metal_exposure", ""),
            "data_source": source,
            "notes": "",
        }

        # Add notes for fallback cases
        if source == "default":
            row["notes"] = "Default fallback values; organism not in curated or heuristic lookup"
        elif source == "heuristic":
            row["notes"] = "Estimated from heuristic organism classification"

        enriched.append(row)

        # Update counts
        organism_counts[organism] = organism_counts.get(organism, 0) + 1
        habitat_counts[row["habitat_type"]] = habitat_counts.get(row["habitat_type"], 0) + 1
        data_source_counts[source] = data_source_counts.get(source, 0) + 1

    # Write output
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline='') as fh:
        fieldnames = [
            "organism",
            "habitat_type",
            "mean_temp_c",
            "mean_ph",
            "metal_exposure",
            "data_source",
            "notes",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in enriched:
            writer.writerow(row)

    logger.info(f"Wrote enriched metadata to {output_csv}")

    return {
        "total_sequences": len(enriched),
        "unique_organisms": len(organism_counts),
        "habitat_distribution": dict(habitat_counts),
        "data_source_distribution": dict(data_source_counts),
        "organisms": dict(organism_counts),
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    input_file = "data/raw/uniprot_sequences_marine.csv"
    output_file = "data/processed/metadata.csv"

    stats = enrich_sequences(input_file, output_file)

    print("\n" + "=" * 70)
    print("ENRICHMENT SUMMARY")
    print("=" * 70)
    print(f"Total sequences enriched: {stats['total_sequences']}")
    print(f"Unique organisms: {stats['unique_organisms']}")
    print("\nHabitat distribution:")
    for habitat, count in sorted(stats['habitat_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {habitat:25s} {count:4d}")
    print("\nData source distribution:")
    for source, count in sorted(stats['data_source_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {source:20s} {count:4d}")
    print("\n" + "=" * 70)
    print(f"Output: {output_file}")
    print("=" * 70)
