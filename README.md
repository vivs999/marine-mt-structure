# Marine metallothionein structure encodes environmental niche

This repository holds the code and data for a study of metallothionein (MT) in
marine animals. MT is a small protein that is rich in cysteine. It binds metal
ions and takes part in the response of a cell to metal stress and oxidative
stress.

The study asks one question. Does the structure of an MT protein carry a
recoverable signal of the environment that the organism lives in?

To answer it, we predicted Zn(2+)-bound MT structures with Boltz-1 for 535
sequences from 272 marine species. We then trained Random Forest models to
predict habitat and five environmental variables from three feature sets: apo
structural geometry, ESM-2 sequence embeddings, and Zn(2+) coordination
geometry from the Boltz predictions.

## What the models show

The primary model is a Random Forest on combined Zn(2+) coordination geometry
and ESM-2 embeddings. We split the data at the level of the organism, so no
species appears in both the training set and the test set. This prevents the
model from recognizing a species it has already seen.

| Target | Metric | Value |
| --- | --- | --- |
| Habitat | Cross-validation accuracy | 90.3% (SD 2.7%) |
| Habitat | Held-out test accuracy | 88.9% |
| Habitat | Held-out test macro F1 | 0.837 |
| Sea surface temperature | Test Pearson r | 0.665 |
| Sea surface temperature | Test R-squared | 0.370 |
| Dissolved oxygen | Test Pearson r | 0.523 |
| Salinity, pH, depth | Test Pearson r | No recoverable signal |

The three feature sets give this ablation on habitat classification:

| Feature set | CV accuracy | Test accuracy |
| --- | --- | --- |
| Apo structural geometry | 85.9% | 79.2% |
| ESM-2 embeddings | 89.9% | 84.7% |
| Zn(2+) geometry with ESM-2 | 90.3% | 88.9% |

The signal is present in the apo structure alone and in the sequence embeddings
alone. Explicit Zn(2+) coordination geometry sharpens it. It does not create
it.

The Zn(2+) geometry in the Boltz predictions is consistent with crystal
references. The mean Zn-S bond length is 2.351 A, against about 2.33 A in
crystal structures. The mean S-Zn-S angle is 111.4 degrees, against 109.5
degrees for an ideal tetrahedron. These values come from 532 structures. They
show that the predictions are chemically plausible. They are not a measurement
of accuracy against experiment.

The signal is selective. It is present for habitat and for thermal variables,
and it is absent for salinity, pH, and depth. This pattern matches the known
biology of MT, which evolved under metal stress and oxidative stress rather
than under osmotic or acid-base pressure.

## What the results do not show

Read these limits before you cite any number above.

- **The models were trained on a subset of the structures.** We predicted 535
  structures, but only 349 records from 185 organisms carry a measured
  Bio-ORACLE environmental label. The models use those 349 records, split into
  277 training rows from 148 species and 72 test rows from 37 species. The
  headline counts of 535 and 272 describe the structural dataset, not the
  modeled cohort.
- **One habitat class is a singleton.** The modeled cohort holds 231
  coastal_estuarine records, 83 open_ocean, 34 polar, and 1 hydrothermal_vent.
  The single hydrothermal vent record cannot appear in both the training set
  and the test set. Treat the habitat task as a three-class problem with a
  singleton attached.
- **The classes are unbalanced.** Coastal_estuarine holds 66% of the cohort.
  Read the macro F1 of 0.837 next to the accuracy of 88.9%.
- **We did not control for phylogeny.** Related species share both a habitat
  and an ancestor. The split at the level of the organism stops data leakage.
  It does not separate an environmental signal from a phylogenetic one.
- **Boltz-1 predictions are not experimental structures.** The geometry checks
  above test plausibility, not accuracy.
- **Bio-ORACLE layers are coarse.** They give a climatological mean for a grid
  cell, not a measurement at the point where the specimen was collected.

## Quickstart

You need Python 3.12 or newer.

```bash
git clone https://github.com/vivs999/marine-mt-structure.git
cd marine-mt-structure
pip install -r requirements.txt

# Primary model: habitat 88.9% test accuracy, SST Pearson r = 0.665
python3 src/train_multitarget.py --features metalbound+esm2
```

The canonical inputs are already in `data/processed/`. You do not need to run
the data collection or the structure prediction stages to reproduce the models.

To reproduce the ablation, run the two baselines:

```bash
python3 src/train_multitarget.py --features structural
python3 src/train_multitarget.py --features esm2
```

Each run writes to `results/models/multitarget/<feature_set>/`. The three
feature sets in the tables above are already there for comparison.

The primary model takes about 10 to 15 minutes on a laptop. The two baselines
are faster. Run every script from the root of the repository, because the
scripts resolve their data paths relative to it.

Expect the last digit of a regression metric to move between runs. Random
Forest sums floating-point values in an order that depends on threads, so
values drift by about 1 part in 10^15. The habitat classification results are
reproducible exactly.

To rebuild the figures, you need R with ggplot2, patchwork, viridis,
RColorBrewer, cowplot, scales, reshape2, dplyr, and readr:

```bash
Rscript src/generate_paper1_figures.R
```

## Repository layout

```
src/                 Pipeline scripts. Run them from the repository root.
notebooks/           ESM-2 embedding generation for Colab.
data/processed/      Canonical inputs for training and figures.
results/models/      Model metrics for the three feature sets.
figures/             Paper figures 1 to 6, as PNG and PDF.
requirements.txt     Python dependencies.
```

The scripts in `src/` form a flat package, because several of them import each
other across pipeline stages.

## Pipeline

The stages chain through CSV files in `data/processed/`. Run them in this
order.

**1. Data collection**

| Script | Purpose |
| --- | --- |
| `ingest_uniprot.py` | Fetch MT sequences from UniProt. |
| `search_new_mt_sequences.py`, `expand_marine_mt_cohort.py` | Extend the cohort. |
| `fetch_obis_occurrences.py` | Get species occurrence records from OBIS. |
| `sample_biooracle.py` | Get environmental layers from Bio-ORACLE v3. |
| `refetch_obis_failed.py`, `refetch_biooracle_delta.py` | Retry failed requests. |
| `enrich_metadata.py`, `enrich_labels_v2.py` | Build the environmental labels. |
| `integrate_enriched_metadata_v2.py`, `integrate_populated.py` | Merge into `integrated_v2.csv`. |

**2. Structure prediction and features**

| Script | Purpose |
| --- | --- |
| `prepare_fasta.py` | Write the FASTA input. |
| `predict_metal_bound_structures.py` | Predict Zn(2+)-bound structures with Boltz-1. |
| `extract_metalbound_features.py` | Extract Zn(2+) coordination geometry features. |
| `batch_extract_features.py`, `feature_extraction.py`, `enhanced_features.py` | Extract apo structural and sequence features. |
| `generate_esm2_embeddings.py` | Compute mean-pooled ESM-2 embeddings. |

**3. Training and figures**

| Script | Purpose |
| --- | --- |
| `train_multitarget.py` | Train and evaluate all models across the six targets. |
| `generate_paper1_figures.R` | Build the paper figures from the model outputs. |
| `generate_paper1_figures.py` | Build the same figures in Python, if you do not have R. |

The other files in `src/` are helper modules. `metadata.py`,
`uniprot_id_utils.py`, and `uniprot_taxa.py` support the data collection
stage. `statistical_analysis.py` holds correlation utilities with Bonferroni
correction. `integrate_enriched_metadata.py` is an earlier version that
`integrate_enriched_metadata_v2.py` replaced.

## Data sources

Every raw input is public.

- Sequences come from [UniProt](https://www.uniprot.org) and NCBI.
- Species occurrence records come from [OBIS](https://obis.org).
- Environmental layers come from [Bio-ORACLE v3](https://www.bio-oracle.org)
  through its ERDDAP server.

The processed files in `data/processed/` were derived from these sources with
the scripts in `src/`.

The file `esm2_embeddings.csv` holds the 2560-dimensional ESM-2 embeddings for
every sequence, and it is 27 MB. The Boltz structure files are too large to
include here. To rebuild them, run `predict_metal_bound_structures.py`.

## Requirements

The core dependencies are pinned in `requirements.txt`.

You need `torch` and `transformers` only if you want to rebuild the ESM-2
embeddings on your own machine. The notebook in `notebooks/` builds them on
Colab instead.

Boltz-1 is installed and run separately for the structure prediction stage.

## Manuscript

A manuscript that reports these results is in preparation. This repository is
the public code and data release that goes with it. We will add the citation
and the preprint link here when they exist.

## Disclosure of AI assistance

AI coding assistants helped to draft the code, the documentation, and the
figure scripts in this repository, under human supervision. The author made and
reviewed every scientific decision. This covers the hypothesis, the choice of
features and targets, the validation design, the interpretation of the results,
and the limits set on each claim. The author treated AI output as a draft and
corrected or rejected it. The scripts in `src/` produced every number in
`results/models/`, and you can regenerate them with the commands above.

## License

The code is released under the MIT License. See `LICENSE`.
