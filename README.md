# Marine metallothionein structure encodes environmental niche

Code and data for a study testing whether the structure of marine
metallothionein (MT) proteins carries a recoverable signal of the environment
in which the organism lives.

Zn(2+)-bound MT structures were predicted with Boltz-1 for 535 sequences across
272 marine species. Random Forest models were then trained to predict habitat
and several environmental variables from three feature sets: apo structural
geometry, ESM-2 sequence embeddings, and Boltz metal-coordination geometry.

## Main results

Primary model: Random Forest on combined metal-coordination geometry and ESM-2
embeddings, evaluated with an organism-level (species-grouped) hold-out split so
no species appears in both training and test sets.

| Target | Metric | Value |
|---|---|---|
| Habitat (4-class) | Cross-validation accuracy | 90.3% (SD 2.7%) |
| Habitat (4-class) | Held-out test accuracy | 88.9% |
| Sea surface temperature | Test Pearson r | 0.665 |
| Sea surface temperature | Test R-squared | 0.370 |
| Dissolved oxygen | Test Pearson r | 0.529 |
| Salinity, pH, depth | - | No recoverable signal |

Feature ablation on habitat classification (held-out test accuracy):

| Feature set | CV accuracy | Test accuracy |
|---|---|---|
| Apo structural geometry | 85.9% | 79.2% |
| ESM-2 embeddings | 89.9% | 84.7% |
| Metal geometry + ESM-2 | 90.3% | 88.9% |

Zn(2+) coordination geometry from the Boltz predictions is consistent with
crystal references: mean Zn-S bond length 2.351 A (crystal ~2.33 A), mean
S-Zn-S angle 111.4 degrees (ideal tetrahedral 109.5 degrees).

The environmental signal is selective: it is present for habitat and thermal
variables but absent for salinity, pH, and depth. This pattern matches the
known biology of MT, which evolved under metal and oxidative stress rather than
osmotic or acid-base pressure.

## Repository layout

```
src/                 Pipeline scripts (flat layout; run from repo root)
notebooks/           ESM-2 embedding generation (Colab)
data/processed/      Canonical inputs for training and figures
results/models/      Trained model metrics for the three feature sets
figures/             Paper figures (Fig 1 to Fig 6), PNG and PDF
requirements.txt     Python dependencies
```

The scripts in `src/` are a flat package because several import each other
across pipeline stages. Run all scripts from the repository root so relative
data paths resolve.

## Pipeline

The pipeline chains through CSV files in `data/processed/`. Stages, in order:

**1. Data collection**
- `ingest_uniprot.py` - fetch MT sequences from UniProt
- `search_new_mt_sequences.py`, `expand_marine_mt_cohort.py` - extend the cohort
- `fetch_obis_occurrences.py` - species occurrence records from OBIS
- `sample_biooracle.py` - environmental layers from Bio-ORACLE v3 (ERDDAP)
- `refetch_obis_failed.py`, `refetch_biooracle_delta.py` - retry helpers
- `enrich_metadata.py`, `enrich_labels_v2.py` - build environmental labels
- `integrate_enriched_metadata.py`, `integrate_enriched_metadata_v2.py`,
  `integrate_populated.py` - merge into `integrated_v2.csv`

**2. Structure prediction and features**
- `prepare_fasta.py` - write FASTA input
- `predict_metal_bound_structures.py` - Boltz-1 Zn(2+)-bound structures
- `extract_metalbound_features.py` - metal-coordination geometry features
- `batch_extract_features.py`, `feature_extraction.py`, `enhanced_features.py` -
  apo structural and sequence features
- `generate_esm2_embeddings.py` or `notebooks/esm2_embeddings.ipynb` - ESM-2
  mean-pooled embeddings

**3. Training**
- `train_multitarget.py` - Random Forest, Ridge/Logistic, and SVR/SVC across all
  six targets, with organism-level grouped cross-validation

**4. Figures**
- `generate_paper1_figures.R` - ggplot2 figures from the model outputs

## Reproducing the main results

The canonical inputs are already in `data/processed/`, so training runs without
rerunning the data-collection or structure-prediction stages.

```bash
pip install -r requirements.txt

# Primary model (habitat 88.9% test, SST r = 0.665):
python3 src/train_multitarget.py --features metalbound+esm2

# Ablation baselines:
python3 src/train_multitarget.py --features structural
python3 src/train_multitarget.py --features esm2
```

Outputs are written to `results/models/multitarget/<feature_set>/`. The three
feature sets reported in the paper are already included there for reference.

To rebuild the figures (requires R with ggplot2, patchwork, viridis,
RColorBrewer, cowplot, scales, reshape2, dplyr, readr):

```bash
Rscript src/generate_paper1_figures.R
```

## Data sources

All raw inputs are public:

- Sequences: UniProt (https://www.uniprot.org) and NCBI
- Species occurrences: OBIS (https://obis.org)
- Environmental layers: Bio-ORACLE v3 (https://www.bio-oracle.org) via ERDDAP

The processed inputs in `data/processed/` were derived from these sources with
the scripts in `src/`. The Boltz-predicted structure files are large and are not
included; regenerate them with `predict_metal_bound_structures.py`.

`esm2_embeddings.csv` (27 MB) holds the 2560-dimensional ESM-2 embeddings for
all sequences.

## Requirements

Python 3.12 or newer. Core dependencies are pinned in `requirements.txt`.
`torch` and `transformers` are optional and only needed to regenerate ESM-2
embeddings locally; the notebook generates them on Colab instead. Boltz-1 is
installed and run separately for the structure-prediction stage.

## Manuscript

A manuscript reporting these results is in preparation. This repository is the
public code and data release accompanying it, and will be updated with a
citation and preprint link once available.

## AI-assisted development disclosure

Portions of the code, documentation, and figure-generation scripts in this
repository were drafted with AI coding assistants under human supervision. All
scientific decisions — hypothesis framing, feature and target selection, the
organism-level validation design, interpretation of results, and the boundaries
of every claim made — were made and reviewed by the author. AI outputs were
treated as drafts subject to correction or rejection. The results in
`results/models/` were produced by running the scripts in `src/`, and can be
regenerated with the commands above.

## License

Code is released under the MIT License (see `LICENSE`).
