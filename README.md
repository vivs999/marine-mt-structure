# Marine metallothionein structure encodes environmental niche

This repository holds the code and data for a study of metallothionein (MT) in
marine animals. MT is a small protein that is rich in cysteine. It binds metal
ions and takes part in the response of a cell to metal stress and oxidative
stress.

The study asks one question. Does the structure of an MT protein carry a
recoverable signal of the environment that the organism lives in?

To answer it, we predicted Zn(2+)-bound MT structures with Boltz-1 for 535
sequences from 252 species. We then trained Random Forest models to predict
habitat and five environmental variables from three feature sets: apo
structural geometry, ESM-2 sequence embeddings, and Zn(2+) coordination
geometry from the Boltz predictions.

The models are fit on the 307 records from 145 marine and diadromous species
that carry a measured Bio-ORACLE environmental label.

## What the models show

Every number below comes from repeated stratified group cross-validation. There
are 10 repeats of 5-fold `StratifiedGroupKFold`. Each fold keeps every species
whole, so no species is ever in both the training and the test side, and each
fold holds a balanced mix of habitat classes. Every record is predicted once per
repeat by a model that never saw its species. We report the mean across repeats
with its standard deviation.

| Target | Metric | Value |
| --- | --- | --- |
| Habitat | Accuracy | 0.870 (SD 0.011) |
| Habitat | Macro F1 | 0.844 (SD 0.011) |
| Sea surface temperature | Pearson r | 0.528 (SD 0.003) |
| Sea surface temperature | R-squared | 0.264 (SD 0.004) |
| Dissolved oxygen | Pearson r | see the note below |
| Salinity, pH, depth | - | No recoverable signal |

Count the recovered signal as one axis, not two. Sea surface temperature and
dissolved oxygen correlate at Pearson r of -0.939 across the cohort, which
follows from the way oxygen solubility depends on temperature. The dissolved
oxygen result is the thermal result in different units.

Habitat performance by class, pooled out of fold across all 307 records:

| Class | Precision | Recall | F1 | n |
| --- | --- | --- | --- | --- |
| coastal_estuarine | 0.87 | 0.97 | 0.91 | 194 |
| open_ocean | 0.92 | 0.73 | 0.82 | 79 |
| polar | 0.93 | 0.76 | 0.84 | 34 |

The three feature sets give this ablation:

| Feature set | Accuracy | Macro F1 | SST Pearson r |
| --- | --- | --- | --- |
| Apo structural geometry | 0.819 (SD 0.010) | 0.782 (SD 0.010) | 0.460 |
| ESM-2 embeddings | 0.868 (SD 0.014) | 0.839 (SD 0.014) | 0.526 |
| Zn(2+) geometry with ESM-2 | 0.870 (SD 0.011) | 0.844 (SD 0.011) | 0.528 |

Read that table carefully, because it does not say what an earlier version of
this README said.

ESM-2 embeddings beat apo structural geometry by a wide margin. Accuracy rises
by 4.9 points and macro F1 by 5.7 points, several times the standard deviation.
That gain is real.

**Explicit Zn(2+) coordination geometry adds nothing that we can measure.**
Adding it to ESM-2 moves accuracy by 0.002, macro F1 by 0.005, and SST
correlation by 0.002, against standard deviations near 0.011. We report this as
a negative result. Predicting the metal-bound structure did not improve
ecological prediction over sequence embeddings alone.

The thermal signal behaves the same way. Apo geometry reaches r = 0.460 and
ESM-2 reaches 0.526, so the embeddings help there too. Adding Zn geometry moves
it to 0.528, which is again nothing.

The Zn(2+) geometry in the Boltz predictions is consistent with crystal
references. The mean Zn-S bond length is 2.351 A, against about 2.33 A in
crystal structures. The mean S-Zn-S angle is 111.4 degrees, against 109.5
degrees for an ideal tetrahedron. These values come from 532 structures. They
show that the predictions are chemically plausible. They are not a measurement
of accuracy against experiment, and the ablation above shows that this
plausibility did not translate into predictive value.

The environmental signal is selective. It is present for habitat and for the
thermal axis, and it is absent for salinity, pH, and depth. This pattern matches
the known biology of MT, which evolved under metal stress and oxidative stress
rather than under osmotic or acid-base pressure. Note that the pH null is weak
evidence, because the pH labels barely vary across the cohort.

## What the results do not show

Read these limits before you cite any number above.

- **A single hold-out split cannot support these numbers, and we no longer use
  one.** An earlier version of this work reported 88.9% accuracy from one
  `GroupShuffleSplit` hold-out. That split groups by species but does not
  balance habitat classes. Across seeds its test set ranged from 43 to 81
  records, its polar class from 3 to 14 records, and its accuracy from 0.750 to
  0.942. The headline moved by 19 points on nothing but the draw. Repeated
  stratified group cross-validation removes that variance, and the standard
  deviation falls from 0.051 to 0.011.
- **We did not control for phylogeny.** Related species share both a habitat and
  an ancestor. Keeping species whole across folds stops data leakage. It does not
  separate an environmental signal from a phylogenetic one. Habitat is close to
  clade-defined in this cohort, so a meaningful share of the accuracy above may
  reflect taxonomy rather than adaptation.
- **The classes are unbalanced.** Coastal_estuarine holds 63% of the cohort.
  Read the macro F1 of 0.844 next to the accuracy of 0.870, and note that open
  ocean recall is 0.73.
- **The cohort is 307 records from 145 species.** That is small. The 95% range
  across repeats is 0.850 to 0.882 for accuracy, and that range describes
  variation between fold assignments, not between studies.
- **One habitat class was dropped.** The dataset holds a single
  hydrothermal_vent record. One record cannot be trained on or predicted, and
  carrying it as a fourth class forces its F1 to 0 and lowers a macro average by
  about 0.21 as an artifact. It is excluded from modelling.
- **Non-marine organisms were removed.** An earlier cohort included freshwater
  fishes, a zebra mussel, a land snail, and a green alga, which take Bio-ORACLE
  values from nearby coastal grid cells. The `origin` column records this, and
  the pipeline now keeps only marine and diadromous organisms.
- **Boltz-1 predictions are not experimental structures.** The geometry checks
  above test plausibility, not accuracy.
- **Bio-ORACLE layers are coarse.** They give a climatological mean for a grid
  cell, not a measurement at the point where the specimen was collected. The
  labels describe the species, not the individual.
- **The pH labels have almost no range.** pH spans 0.521 units across the whole
  cohort, from 7.75 to 8.27. Read the pH result as a statement about the label,
  not about MT.

## Quickstart

You need Python 3.12 or newer.

```bash
git clone https://github.com/vivs999/marine-mt-structure.git
cd marine-mt-structure
pip install -r requirements.txt

# Headline numbers, repeated stratified group cross-validation:
python3 src/robust_eval.py metalbound+esm2
```

The canonical inputs are already in `data/processed/`. You do not need to run
the data collection or the structure prediction stages to reproduce the models.

To reproduce the ablation, run the two other feature sets:

```bash
python3 src/robust_eval.py structural
python3 src/robust_eval.py esm2
```

`src/train_multitarget.py` runs the older single hold-out design across all six
targets and all three model families. It is kept because it produces the
per-target artifacts in `results/models/`, and because its cross-validation
column is still informative. Do not take its held-out numbers as headline
results, for the reason given in the limits above.

```bash
python3 src/train_multitarget.py --features metalbound+esm2
```

Run every script from the root of the repository, because the scripts resolve
their data paths relative to it.

A full `robust_eval.py` run takes about 10 to 20 minutes on a laptop. Model
fitting is CPU-bound. There is no GPU path, because scikit-learn tree ensembles
run on the CPU on every platform. The one GPU workload here is ESM-2 embedding
generation, and `src/generate_esm2_embeddings.py` already selects CUDA or Metal
when either is present.

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

## The data file

`data/processed/integrated_v2.csv` holds all 535 sequences, so both the full set
and the modelled cohort can be rebuilt from it. Three columns matter for cohort
selection.

`organism` is a normalized binomial. An earlier version of this file carried 11
species under two spellings each, such as `Gadus morhua` and `Gadus morhua
(Atlantic cod)`. Because the split grouped on that string, three species
appeared in both the training and the test set. Grouping now uses this
normalized column.

`organism_verbatim` preserves the original string from UniProt.

`origin` records whether the organism is marine, diadromous, freshwater,
terrestrial, or not an animal. `load_data` keeps marine and diadromous rows.

`sst_mean_c_status` records whether the sea surface temperature label is a real
Bio-ORACLE sample or a habitat-derived default. Only `biooracle_sampled` rows
enter the cohort.

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
| `robust_eval.py` | Repeated stratified group CV. Produces the headline numbers. |
| `train_multitarget.py` | Single hold-out across all six targets. Produces the per-target artifacts. |
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
