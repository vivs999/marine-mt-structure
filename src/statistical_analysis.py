"""Statistical analysis utilities for the marine metallothionein pipeline.

Functions:
    pearson_correlations   - pairwise Pearson r with Bonferroni correction
    spearman_correlations  - pairwise Spearman r with Bonferroni correction
    anova_by_habitat       - one-way ANOVA for each feature across habitat types
    summarise_features     - descriptive statistics stratified by habitat
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import f_oneway
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


def _require_scipy() -> None:
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy and pandas are required for statistical_analysis")


STRUCTURAL_FEATURES = [
    "cysteine_density",
    "mean_ss_distance",
    "rg_cysteine_sulfur",
    "n_cys_sulfur_atoms",
]

ENVIRONMENTAL_VARS = [
    "mean_temp_c",
    "mean_ph",
]


def to_dataframe(rows: List[Dict]) -> "pd.DataFrame":
    """Convert a list of dicts (integrated dataset) to a pandas DataFrame.

    Numeric columns are coerced; non-convertible values become NaN.
    """
    _require_scipy()
    df = pd.DataFrame(rows)
    numeric_cols = STRUCTURAL_FEATURES + ENVIRONMENTAL_VARS + [
        "cysteine_count", "length"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pearson_correlations(
    df: "pd.DataFrame",
    features: Optional[List[str]] = None,
    env_vars: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> "pd.DataFrame":
    """Compute Pearson r between structural features and environmental variables.

    Returns a DataFrame with columns: feature, env_var, r, p_raw, p_bonferroni,
    significant (bool at *alpha* after Bonferroni correction).
    """
    _require_scipy()
    features = [f for f in (features or STRUCTURAL_FEATURES) if f in df.columns]
    env_vars = [v for v in (env_vars or ENVIRONMENTAL_VARS) if v in df.columns]

    records = []
    for feat in features:
        for env in env_vars:
            mask = df[feat].notna() & df[env].notna()
            x = df.loc[mask, feat]
            y = df.loc[mask, env]
            if len(x) < 3:
                logger.warning("Too few observations for %s vs %s (n=%d)", feat, env, len(x))
                continue
            r, p = stats.pearsonr(x, y)
            records.append({"feature": feat, "env_var": env, "r": r, "p_raw": p, "n": int(len(x))})

    result = pd.DataFrame(records)
    if result.empty:
        return result
    n_tests = len(result)
    result["p_bonferroni"] = (result["p_raw"] * n_tests).clip(upper=1.0)
    result["significant"] = result["p_bonferroni"] < alpha
    return result


def spearman_correlations(
    df: "pd.DataFrame",
    features: Optional[List[str]] = None,
    env_vars: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> "pd.DataFrame":
    """Compute Spearman rank correlation between structural features and env vars.

    Same output schema as :func:`pearson_correlations`.
    """
    _require_scipy()
    features = [f for f in (features or STRUCTURAL_FEATURES) if f in df.columns]
    env_vars = [v for v in (env_vars or ENVIRONMENTAL_VARS) if v in df.columns]

    records = []
    for feat in features:
        for env in env_vars:
            mask = df[feat].notna() & df[env].notna()
            x = df.loc[mask, feat]
            y = df.loc[mask, env]
            if len(x) < 3:
                continue
            r, p = stats.spearmanr(x, y)
            records.append({"feature": feat, "env_var": env, "r": r, "p_raw": p, "n": int(len(x))})

    result = pd.DataFrame(records)
    if result.empty:
        return result
    n_tests = len(result)
    result["p_bonferroni"] = (result["p_raw"] * n_tests).clip(upper=1.0)
    result["significant"] = result["p_bonferroni"] < alpha
    return result


def anova_by_habitat(
    df: "pd.DataFrame",
    features: Optional[List[str]] = None,
    habitat_col: str = "habitat_type",
) -> "pd.DataFrame":
    """One-way ANOVA for each structural feature across habitat types.

    Returns a DataFrame with columns: feature, F, p, significant.
    """
    _require_scipy()
    features = [f for f in (features or STRUCTURAL_FEATURES) if f in df.columns]
    if habitat_col not in df.columns:
        raise ValueError(f"Habitat column '{habitat_col}' not found in DataFrame")

    records = []
    for feat in features:
        groups = []
        for htype, grp in df.groupby(habitat_col):
            vals = grp[feat].dropna().values
            if len(vals) >= 2:
                groups.append(vals)
        if len(groups) < 2:
            logger.warning("Fewer than 2 groups with data for %s; skipping ANOVA", feat)
            continue
        F, p = f_oneway(*groups)
        records.append({"feature": feat, "F": F, "p": p, "significant": p < 0.05})

    return pd.DataFrame(records)


def summarise_features(
    df: "pd.DataFrame",
    features: Optional[List[str]] = None,
    habitat_col: str = "habitat_type",
) -> "pd.DataFrame":
    """Compute descriptive statistics for each feature, stratified by habitat type.

    Returns a long-form DataFrame with columns: habitat_type, feature,
    mean, std, min, q25, median, q75, max, n.
    """
    _require_scipy()
    features = [f for f in (features or STRUCTURAL_FEATURES) if f in df.columns]

    records = []
    for feat in features:
        if habitat_col in df.columns:
            grouped = df.groupby(habitat_col)[feat]
        else:
            grouped = {"all": df[feat]}

        for htype, series in grouped:
            vals = series.dropna()
            if len(vals) == 0:
                continue
            records.append({
                "habitat_type": htype,
                "feature": feat,
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "q25": float(vals.quantile(0.25)),
                "median": float(vals.median()),
                "q75": float(vals.quantile(0.75)),
                "max": float(vals.max()),
                "n": int(len(vals)),
            })

    # Also compute overall (all habitats combined)
    for feat in features:
        vals = df[feat].dropna()
        if len(vals) == 0:
            continue
        records.append({
            "habitat_type": "overall",
            "feature": feat,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "q25": float(vals.quantile(0.25)),
            "median": float(vals.median()),
            "q75": float(vals.quantile(0.75)),
            "max": float(vals.max()),
            "n": int(len(vals)),
        })

    return pd.DataFrame(records)
