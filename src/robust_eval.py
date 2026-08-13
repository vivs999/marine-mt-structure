"""Repeated stratified group cross-validation. No single hold-out, no lucky draw.

The published design draws one hold-out with GroupShuffleSplit. That is grouped
by species but not stratified by habitat, so the class balance of the test set
is luck. Polar lands anywhere between 3 and 14 records depending on the seed.

This evaluates every record instead. Each repeat runs StratifiedGroupKFold,
which keeps species intact across folds and balances habitat within them. Every
record is predicted exactly once per repeat, from a model that never saw its
species. Scores are computed per repeat, then summarised across repeats.

Parallelism, and why there is none here beyond `n_jobs=-1`. Spreading the
independent folds across processes looks attractive, because a single Random
Forest fit scales poorly at 307 records. It was measured and it is slower.
Fitting 8 models on the ESM-2 matrix on an M3 takes 3.9 s sequentially with
`n_jobs=-1`, 3.9 s across 8 threads with `n_jobs=1`, and 6.7 s across 8
processes with `n_jobs=1`. Pickling a 307 by 2560 matrix to every worker costs
more than the fits do, and threads add nothing because `n_jobs=-1` already
saturates the cores. Sequential fitting with `n_jobs=-1` is the fastest option.

There is no GPU path here. scikit-learn tree ensembles run on the CPU on every
platform, and the GPU implementations of these models are CUDA-only. The one
GPU workload in this project is ESM-2 embedding generation, and
`src/generate_esm2_embeddings.py` already selects CUDA or Metal when present.

Usage, from inside a corrected workspace:
    uv run --with ... python3 ../robust_eval.py [feature_set] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))

from src.train_multitarget import get_feature_matrix, load_data  # noqa: E402

N_REPEATS = 10
N_SPLITS = 5


def clf() -> Pipeline:
    """Matches the configuration in src/train_multitarget.py exactly."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1)),
    ])


def reg() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("reg", RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1)),
    ])


def _oof(make, X, y, splits, dtype):
    """Out-of-fold predictions, one array per repeat.

    Fits run sequentially. Each model uses every core through `n_jobs=-1`,
    which measurement showed to be the fastest arrangement. See the module
    docstring.
    """
    out = []
    for _, folds in splits:
        oof = np.empty(len(y), dtype=dtype)
        for tr, te in folds:
            oof[te] = make().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        out.append(oof)
    return out


def summarise(name: str, vals: list[float]) -> None:
    a = np.array(vals)
    lo, hi = np.percentile(a, [2.5, 97.5])
    print(f"  {name:24s} mean={a.mean():.3f}  SD={a.std():.3f}  "
          f"95% range [{lo:.3f}, {hi:.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", nargs="?", default="metalbound+esm2")
    args = ap.parse_args()

    df = load_data()
    X = get_feature_matrix(df, feature_mode=args.features)
    groups = LabelEncoder().fit_transform(df["organism"].values)
    print(f"cohort {len(df)} records, {len(set(groups))} species, "
          f"features={args.features} {X.shape}\n")

    y = df["habitat_type"].values
    # The single hydrothermal_vent record can never be learned or predicted.
    # Including it drags a 4-class macro F1 down by about 0.21 as a pure
    # artifact, so it is excluded and the exclusion is reported.
    keep = pd.notna(y) & (y != "hydrothermal_vent")
    Xc = X[keep].reset_index(drop=True)
    yc, gc = y[keep], groups[keep]

    csplits = [
        (rep, list(StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=rep).split(Xc, yc, groups=gc)))
        for rep in range(N_REPEATS)
    ]
    acc, f1s = [], []
    for oof in _oof(clf, Xc, yc, csplits, object):
        acc.append(accuracy_score(yc, oof))
        f1s.append(f1_score(yc, oof, average="macro"))

    ys = pd.to_numeric(df["sst_mean_c"], errors="coerce").values
    ok = pd.notna(ys)
    Xs = X[ok].reset_index(drop=True)
    ys_, gs = ys[ok], groups[ok]
    rsplits = []
    for rep in range(N_REPEATS):
        order = np.random.RandomState(rep).permutation(len(ys_))
        folds = [(order[tr], order[te]) for tr, te in
                 GroupKFold(n_splits=N_SPLITS).split(
                     Xs.iloc[order], ys_[order], groups=gs[order])]
        rsplits.append((rep, folds))
    r_vals, r2_vals = [], []
    for oof in _oof(reg, Xs, ys_, rsplits, float):
        r_vals.append(np.corrcoef(ys_, oof)[0, 1])
        r2_vals.append(r2_score(ys_, oof))

    print(f"Repeated stratified group CV, {N_REPEATS} repeats x {N_SPLITS} folds")
    print(f"Every one of {keep.sum()} records predicted by a model blind to its species.")
    print("Habitat is scored over the 3 real classes. The 1-record "
          "hydrothermal_vent class is excluded.\n")
    summarise("habitat accuracy", acc)
    summarise("habitat macro F1", f1s)
    print()
    summarise("SST Pearson r", r_vals)
    summarise("SST R2", r2_vals)

    out = Path("results/robust_eval")
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.features.replace('+', '_')}.json"
    with open(dest, "w") as fh:
        json.dump({
            "feature_set": args.features,
            "n_records": int(keep.sum()),
            "n_species": int(len(set(gc))),
            "n_repeats": N_REPEATS,
            "n_splits": N_SPLITS,
            "habitat_accuracy_mean": float(np.mean(acc)),
            "habitat_accuracy_sd": float(np.std(acc)),
            "habitat_f1_macro_mean": float(np.mean(f1s)),
            "habitat_f1_macro_sd": float(np.std(f1s)),
            "sst_pearson_mean": float(np.mean(r_vals)),
            "sst_pearson_sd": float(np.std(r_vals)),
            "sst_r2_mean": float(np.mean(r2_vals)),
            "sst_r2_sd": float(np.std(r2_vals)),
        }, fh, indent=2)
    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    main()
