#!/usr/bin/env python3
"""
Phase 3 - Multi-target ML training.

Trains three model families across six biological targets:
  Classification: habitat_type (4-class)
  Regression:     sst_mean_c, salinity_mean_psu, ph_mean, depth_mean_m, do_mean_mlL

Models: RandomForest (primary), Ridge/LogReg (linear baseline), SVR/SVC (nonlinear baseline)

Validation: organism-level grouped 5-fold CV to prevent species leakage.

Inputs:
  data/processed/integrated_v2.csv      (structural features + metadata)
  data/processed/esm2_embeddings.csv    (ESM-2 language model embeddings, optional)

Feature modes (--features flag):
  structural  - 47 geometry/sequence features from ESMFold apo structures (default)
  esm2        - mean-pooled ESM-2 embeddings (biologically meaningful for IDPs)

Outputs:
  results/models/multitarget/{structural,esm2}/{target}/{model}_metrics.json
  results/models/multitarget/{structural,esm2}/summary_table.csv
"""

from __future__ import annotations
import argparse, csv, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_absolute_error,
    mean_squared_error, make_scorer,
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# Paths
INTEGRATED_V2   = Path("data/processed/integrated_v2.csv")
INTEGRATED_V1   = Path("data/processed/integrated_populated.csv")
ESM2_EMBEDDINGS    = Path("data/processed/esm2_embeddings.csv")
METALBOUND_FEATS   = Path("data/processed/metalbound_features.csv")
OUT_BASE           = Path("results/models/multitarget")

# Targets
TARGETS = {
    "habitat_type":      {"type": "classification"},
    "sst_mean_c":        {"type": "regression"},
    "salinity_mean_psu": {"type": "regression"},
    "ph_mean":           {"type": "regression"},
    "depth_mean_m":      {"type": "regression"},
    "do_mean_mlL":       {"type": "regression"},
}

# Structural feature columns
# These match the output of src.feature_extraction + src.enhanced_features.
FEATURE_COLS = [
    # Cysteine sulfur geometry
    "n_cys_sulfur_atoms", "mean_ss_distance", "rg_cysteine_sulfur",
    "cysteine_density",
    # Log / squared transforms (nonlinear)
    "log_mean_ss_distance", "mean_ss_distance_sq",
    "log_rg_cysteine_sulfur", "rg_cysteine_sulfur_sq",
    "ss_rg_ratio_sq", "compactness_index", "normalized_rg", "normalized_ss",
    "density_weighted_ss", "density_weighted_rg",
    # Cysteine motifs (density)
    "cxc_motif_density", "cxxc_motif_density", "cc_motif_density",
    "kc_motif_density", "sc_motif_density", "pc_motif_density",
    # Cysteine spacing
    "cys_spacing_mean", "cys_spacing_std", "cys_spacing_min",
    "cys_spacing_max", "cys_spacing_cv", "max_consecutive_cys",
    # Cysteine positional bias
    "cys_n_term_bias", "cys_c_term_bias", "cys_center_fraction",
    # Cysteine local context
    "cys_context_hydrophobic", "cys_context_charged",
    "cys_context_polar", "cys_context_aromatic",
    # Amino acid composition
    "positive_fraction", "negative_fraction", "charged_fraction",
    "hydrophobic_fraction", "polar_fraction", "aromatic_fraction",
    "small_fraction", "proline_fraction", "glycine_fraction",
    "lysine_fraction", "arginine_fraction", "serine_fraction",
    "net_charge_density", "charge_asymmetry",
]


def load_metalbound_features(df: pd.DataFrame) -> pd.DataFrame:
    """Load Boltz-predicted metal-bound coordination geometry features."""
    if not METALBOUND_FEATS.exists():
        raise FileNotFoundError(
            f"{METALBOUND_FEATS} not found. "
            "Run: uv run --python 3.12 --with boltz python3 src/predict_metal_bound_structures.py"
            " then: uv run --python 3.12 python3 src/extract_metalbound_features.py"
        )
    mb = pd.read_csv(METALBOUND_FEATS)
    mb_cols = [c for c in mb.columns if c.startswith("mb_")]
    print(f"  Metal-bound features: {mb.shape[0]} structures x {len(mb_cols)} features")

    mb_dedup = mb[["uniprot_id"] + mb_cols].drop_duplicates(subset=["uniprot_id"])
    aligned = df[["uniprot_id"]].reset_index(drop=True).merge(mb_dedup, on="uniprot_id", how="left")
    n_missing = aligned[mb_cols[0]].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing}/{len(aligned)} rows have no metal-bound features - will be imputed")
    return aligned[mb_cols].reset_index(drop=True)


def load_esm2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Load ESM-2 embeddings and align to df by uniprot_id. Returns feature DataFrame."""
    if not ESM2_EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"{ESM2_EMBEDDINGS} not found. "
            "Run the Colab notebook (notebooks/esm2_embeddings.ipynb), download the output, "
            "and place it at data/processed/esm2_embeddings.csv"
        )
    emb = pd.read_csv(ESM2_EMBEDDINGS, index_col="uniprot_id")
    print(f"  ESM-2 embeddings: {emb.shape[0]} sequences x {emb.shape[1]} dims")
    emb_cols = [c for c in emb.columns if c.startswith("esm2_")]

    emb_dedup = emb[emb_cols].reset_index().drop_duplicates(subset=["uniprot_id"])
    aligned = df[["uniprot_id"]].reset_index(drop=True).merge(
        emb_dedup, on="uniprot_id", how="left"
    )
    n_missing = aligned[emb_cols[0]].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing}/{len(aligned)} rows have no ESM-2 embedding - will be imputed")
    return aligned[emb_cols].reset_index(drop=True)


def load_data(real_only: bool = True) -> pd.DataFrame:
    path = INTEGRATED_V2 if INTEGRATED_V2.exists() else INTEGRATED_V1
    print(f"Loading data from {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape (raw): {df.shape}")

    if real_only:
        # Keep only rows where the primary environmental label (SST) is a real
        # Bio-ORACLE measurement, not a habitat-derived default.  This gives the
        # canonical 349-row dataset used for all primary Paper 1 results.
        if "sst_mean_c_status" in df.columns:
            real_mask = df["sst_mean_c_status"] == "biooracle_sampled"
            n_before = len(df)
            df = df[real_mask].reset_index(drop=True)
            print(f"  Filtered to biooracle_sampled SST rows: {len(df)}/{n_before} "
                  f"(dropped {n_before - len(df)} non-sampled rows)")
        else:
            print("  WARNING: sst_mean_c_status not found - cannot filter. Using all rows.")

    print(f"  Shape (final): {df.shape}")
    return df


def get_feature_matrix(df: pd.DataFrame, feature_mode: str = "structural") -> pd.DataFrame:
    if feature_mode == "esm2":
        return load_esm2_features(df)
    if feature_mode == "metalbound":
        return load_metalbound_features(df)
    if feature_mode == "metalbound+esm2":
        mb  = load_metalbound_features(df)
        emb = load_esm2_features(df)
        return pd.concat([mb.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature cols not found: {missing_cols[:5]}...")
    print(f"  Features available: {len(available)}")
    return df[available].copy()


def pearson_r_scorer(y_true, y_pred):
    from scipy.stats import pearsonr
    r, _ = pearsonr(y_true, y_pred)
    return r


PEARSON_SCORER = make_scorer(pearson_r_scorer)


def make_holdout_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple:
    """Return (X_train, X_test, df_train, df_test) from pre-computed organism-level split."""
    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        df.iloc[train_idx].copy(),
        df.iloc[test_idx].copy(),
    )


def _save_predictions(
    y_true,
    y_pred,
    df_test_slice: "pd.DataFrame",
    target_dir: Path,
    model_name: str,
) -> None:
    uid_col = "uniprot_id" if "uniprot_id" in df_test_slice.columns else None
    rows = {
        "y_true": list(y_true),
        "y_pred": list(y_pred),
    }
    if uid_col:
        rows["uniprot_id"] = list(df_test_slice[uid_col])
    pd.DataFrame(rows).to_csv(
        target_dir / f"{model_name}_predictions.csv", index=False
    )


def train_target(
    target: str,
    task_type: str,
    df: pd.DataFrame,
    X: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path,
) -> list:
    """Train all three models for one target using pre-computed organism-level hold-out split."""

    fallback = {
        "sst_mean_c":        "temperature_c",
        "salinity_mean_psu": "salinity_psu",
        "ph_mean":           "ph",
        "depth_mean_m":      "depth_m",
    }

    col = target if target in df.columns else fallback.get(target)
    if col is None or col not in df.columns:
        print(f"  SKIP {target}: column not found")
        return []

    # Apply hold-out split
    X_train_full = X.iloc[train_idx].copy()
    X_test_full  = X.iloc[test_idx].copy()
    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    # Groups for CV (organism-level, training set only)
    le_groups = LabelEncoder()
    g_train_raw = le_groups.fit_transform(df_train["organism"].values)

    # Target vectors
    y_train_raw = df_train[col].copy()
    y_test_raw  = df_test[col].copy()

    if task_type == "regression":
        train_mask = pd.to_numeric(y_train_raw, errors="coerce").notna()
        test_mask  = pd.to_numeric(y_test_raw,  errors="coerce").notna()
        # Additionally exclude rows where the label is a habitat-derived default,
        # not an actual Bio-ORACLE measurement.
        status_col = f"{col}_status"
        if status_col not in df_train.columns:
            status_col = f"{target}_status"
        if status_col in df_train.columns:
            real_train = ~df_train[status_col].astype(str).str.startswith("habitat_default")
            real_test  = ~df_test[status_col].astype(str).str.startswith("habitat_default")
            train_mask = train_mask & real_train.values
            test_mask  = test_mask  & real_test.values
    else:
        train_mask = y_train_raw.notna() & (y_train_raw != "")
        test_mask  = y_test_raw.notna()  & (y_test_raw  != "")

    X_train = X_train_full[train_mask.values].copy()
    y_train = y_train_raw[train_mask].copy()
    g_train = g_train_raw[train_mask.values]
    X_test  = X_test_full[test_mask.values].copy()
    y_test  = y_test_raw[test_mask].copy()

    if task_type == "regression":
        y_train = pd.to_numeric(y_train, errors="coerce")
        y_test  = pd.to_numeric(y_test,  errors="coerce")

    n_train = len(y_train)
    n_test  = len(y_test)
    n_species_train = len(np.unique(g_train))
    print(f"\n  [{target}] train={n_train}, test={n_test}, "
          f"species_train={n_species_train}, task={task_type}")

    if n_train < 20:
        print(f"  SKIP {target}: too few training samples ({n_train})")
        return []

    # Models
    if task_type == "classification":
        models = {
            "random_forest": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=None,
                    min_samples_leaf=2, random_state=42, n_jobs=-1,
                )),
            ]),
            "logistic_regression": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, C=1.0, random_state=42, n_jobs=-1,
                )),
            ]),
            "svc": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
            ]),
        }
        cv = StratifiedGroupKFold(n_splits=min(5, n_species_train))
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
        cv_kwargs = dict(cv=cv, groups=g_train, scoring=scoring,
                         return_train_score=False, error_score="raise")

    else:
        models = {
            "random_forest": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("reg", RandomForestRegressor(
                    n_estimators=300, max_depth=None,
                    min_samples_leaf=2, random_state=42, n_jobs=-1,
                )),
            ]),
            "ridge": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ]),
            "svr": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
                ("reg", SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)),
            ]),
        }
        cv = GroupKFold(n_splits=min(5, n_species_train))
        scoring = {
            "r2":      "r2",
            "mae":     "neg_mean_absolute_error",
            "rmse":    "neg_root_mean_squared_error",
            "pearson": PEARSON_SCORER,
        }
        cv_kwargs = dict(cv=cv, groups=g_train, scoring=scoring,
                         return_train_score=False, error_score="raise")

    target_dir = out_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name, pipe in models.items():
        print(f"    {model_name}...", end=" ", flush=True)
        try:
            # CV on training set only
            cv_results = cross_validate(pipe, X_train, y_train, **cv_kwargs)

            # Refit on full training set, evaluate ONCE on held-out test set
            pipe.fit(X_train, y_train)

            if task_type == "classification":
                cv_metrics = {
                    "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
                    "cv_accuracy_std":  float(np.std(cv_results["test_accuracy"])),
                    "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
                    "cv_f1_macro_std":  float(np.std(cv_results["test_f1_macro"])),
                }
                if n_test > 0:
                    y_pred = pipe.predict(X_test)
                    test_metrics = {
                        "test_accuracy": float(accuracy_score(y_test, y_pred)),
                        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                    }
                    _save_predictions(y_test, y_pred, df_test[test_mask.values], target_dir, model_name)
                else:
                    test_metrics = {}
                print(f"cv_acc={cv_metrics['cv_accuracy_mean']:.3f} "
                      f"test_acc={test_metrics.get('test_accuracy', float('nan')):.3f}")
                metrics = {**cv_metrics, **test_metrics}

            else:
                cv_metrics = {
                    "cv_r2_mean":      float(np.mean(cv_results["test_r2"])),
                    "cv_r2_std":       float(np.std(cv_results["test_r2"])),
                    "cv_mae_mean":     float(-np.mean(cv_results["test_mae"])),
                    "cv_rmse_mean":    float(-np.mean(cv_results["test_rmse"])),
                    "cv_pearson_mean": float(np.mean(cv_results["test_pearson"])),
                }
                if n_test > 0:
                    y_pred = pipe.predict(X_test)
                    from scipy.stats import pearsonr
                    test_r, _ = pearsonr(y_test, y_pred) if len(y_test) > 1 else (float("nan"), None)
                    test_metrics = {
                        "test_r2":      float(r2_score(y_test, y_pred)),
                        "test_mae":     float(mean_absolute_error(y_test, y_pred)),
                        "test_rmse":    float(np.sqrt(mean_squared_error(y_test, y_pred))),
                        "test_pearson": float(test_r),
                    }
                    _save_predictions(y_test, y_pred, df_test[test_mask.values], target_dir, model_name)
                else:
                    test_metrics = {}
                print(f"cv_R²={cv_metrics['cv_r2_mean']:.3f} "
                      f"test_R²={test_metrics.get('test_r2', float('nan')):.3f} "
                      f"test_r={test_metrics.get('test_pearson', float('nan')):.3f}")
                metrics = {**cv_metrics, **test_metrics}

            metrics.update({
                "target": target,
                "model":  model_name,
                "n_train": int(n_train),
                "n_test":  int(n_test),
                "n_species_train": int(n_species_train),
                "task_type": task_type,
            })

            with open(target_dir / f"{model_name}_metrics.json", "w") as fh:
                json.dump(metrics, fh, indent=2)

            # Feature importances (RF only)
            if model_name == "random_forest":
                step = pipe.named_steps.get("clf") or pipe.named_steps.get("reg")
                if hasattr(step, "feature_importances_"):
                    fi_rows = sorted(
                        zip(X_train.columns.tolist(), step.feature_importances_),
                        key=lambda x: -x[1],
                    )
                    with open(target_dir / "random_forest_feature_importance.csv", "w", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["feature", "importance"])
                        w.writerows(fi_rows)

            summary_rows.append(metrics)

        except Exception as e:
            print(f"ERROR: {e}")
            summary_rows.append({"target": target, "model": model_name, "error": str(e)})

    return summary_rows


def make_organism_split(
    df: pd.DataFrame, X: pd.DataFrame, out_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """80/20 organism-level hold-out split. Saves train/test UniProt IDs to disk."""
    le = LabelEncoder()
    groups = le.fit_transform(df["organism"].values)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, groups=groups))

    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = df.iloc[train_idx]["uniprot_id"].tolist()
    test_ids  = df.iloc[test_idx]["uniprot_id"].tolist()
    with open(out_dir / "train_ids.txt", "w") as f:
        f.write("\n".join(train_ids))
    with open(out_dir / "test_ids.txt", "w") as f:
        f.write("\n".join(test_ids))

    n_train_species = len(np.unique(groups[train_idx]))
    n_test_species  = len(np.unique(groups[test_idx]))
    print(f"Hold-out split: {len(train_idx)} train ({n_train_species} species) / "
          f"{len(test_idx)} test ({n_test_species} species)")
    print(f"  train_ids -> {out_dir}/train_ids.txt")
    print(f"  test_ids  -> {out_dir}/test_ids.txt")

    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        choices=["structural", "esm2", "metalbound", "metalbound+esm2"],
        default="structural",
        help=(
            "Feature set to use:\n"
            "  structural     - 47 apo geometry features (baseline, biologically weak)\n"
            "  esm2           - ESM-2 mean-pooled embeddings (sequence-evolutionary)\n"
            "  metalbound     - Boltz Zn²⁺-bound coordination geometry (explicit metal)\n"
            "  metalbound+esm2- Combined: metal geometry + ESM-2 embeddings (recommended)"
        )
    )
    args = parser.parse_args()

    OUT_DIR = OUT_BASE / args.features
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Feature mode: {args.features}")
    print(f"Output dir:   {OUT_DIR}\n")

    df = load_data()
    X = get_feature_matrix(df, feature_mode=args.features)

    # Create organism-level 80/20 hold-out split ONCE before any training
    train_idx, test_idx = make_organism_split(df, X, OUT_DIR)

    all_summary = []
    for target, cfg in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"TARGET: {target} ({cfg['type']})")
        rows = train_target(target, cfg["type"], df, X, train_idx, test_idx, OUT_DIR)
        if rows:
            all_summary.extend(rows)

    # Write master summary table
    if all_summary:
        all_keys = sorted({k for r in all_summary for k in r})
        with open(OUT_DIR / "summary_table.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_summary)
        print(f"\nSummary table -> {OUT_DIR}/summary_table.csv")

    # Print clean summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for row in all_summary:
        if "error" in row:
            print(f"  {row['target']:20s} {row['model']:25s} ERROR: {row['error'][:50]}")
        elif row.get("task_type") == "classification":
            print(f"  {row['target']:20s} {row['model']:25s} "
                  f"cv_acc={row.get('cv_accuracy_mean', float('nan')):.3f}  "
                  f"test_acc={row.get('test_accuracy', float('nan')):.3f}  "
                  f"test_f1={row.get('test_f1_macro', float('nan')):.3f}")
        else:
            print(f"  {row['target']:20s} {row['model']:25s} "
                  f"cv_R²={row.get('cv_r2_mean', float('nan')):.3f}  "
                  f"test_R²={row.get('test_r2', float('nan')):.3f}  "
                  f"test_r={row.get('test_pearson', float('nan')):.3f}")


if __name__ == "__main__":
    main()
