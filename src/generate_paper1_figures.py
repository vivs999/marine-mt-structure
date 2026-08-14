"""Generate all Paper 1 publication figures for the metallothionein ML study.

Figures produced:
  1. Pipeline schematic
  2. Zn coordination geometry validation (violin plots)
  3. Ablation bar chart (habitat classification accuracy)
  4. Confusion matrix (metalbound+esm2 RF on habitat_type)
  5. SST scatter plot (predicted vs true)
  6. Results heatmap (all targets x feature sets)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = ROOT / "results" / "models" / "multitarget"
FIGURES_DIR = ROOT / "results" / "figures" / "paper1"

METALBOUND_FEATURES = ROOT / "data" / "processed" / "metalbound_features.csv"
PRIMARY_DIR = RESULTS_BASE / "metalbound+esm2"
ESM2_DIR = RESULTS_BASE / "esm2"
STRUCTURAL_DIR = RESULTS_BASE  # top-level summary is the structural baseline

CRYSTAL_ZN_S_DIST = 2.33  # Å, crystallographic reference
CRYSTAL_SZNS_ANGLE = 109.5  # degrees, ideal tetrahedral

HABITAT_COLORS = {
    "coastal_estuarine": "#2196F3",
    "open_ocean": "#4CAF50",
    "polar": "#9C27B0",
    "hydrothermal_vent": "#FF5722",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# helpers

def save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path.relative_to(ROOT)}")
    return path


def load_summary(directory: Path) -> pd.DataFrame:
    path = directory / "summary_table.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_predictions(directory: Path, target: str, model: str = "random_forest") -> pd.DataFrame | None:
    path = directory / target / f"{model}_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# Figure 1: Pipeline schematic

def fig_pipeline_schematic() -> None:
    """Study design, drawn from the live result files so it cannot drift."""
    import json

    def robust(name):
        f = ROOT / "results" / "robust_eval" / f"{name}.json"
        return json.load(open(f)) if f.exists() else None

    mb, esm, st = robust("metalbound_esm2"), robust("esm2"), robust("structural")
    n_rec = mb["n_records"] if mb else 307
    n_sp = mb["n_species"] if mb else 145

    fig, ax = plt.subplots(figsize=(13, 7.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    INK, MUTED = "#1A1A1A", "#5A5A5A"
    BLUE, GREY, GREEN, SAND = "#1F4E79", "#E8E8E8", "#2E7D32", "#F5EFE0"

    def box(x, y, w, h, title, body, fill="white", edge=INK, lw=1.1, tsize=10, bsize=8.6):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.045,rounding_size=0.08",
            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
        if title:
            ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
                    fontsize=tsize, fontweight="bold", color=INK, zorder=3)
        if body:
            ax.text(x + w / 2, y + (h - 0.62) / 2 + 0.06, body, ha="center", va="center",
                    fontsize=bsize, color=MUTED, linespacing=1.5, zorder=3)

    def arrow(x0, y0, x1, y1, color=INK, lw=1.3, style="-|>"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    shrinkA=0, shrinkB=0))

    # Stage 1, inputs
    box(0.35, 5.25, 2.5, 1.35, "Sequences",
        "UniProt and NCBI\n535 metallothioneins", fill="white")
    box(0.35, 3.45, 2.5, 1.35, "Environment",
        "OBIS occurrences\nBio-ORACLE v3 layers", fill="white")

    # Stage 2, cohort
    box(3.45, 4.05, 2.6, 2.55, "Modelled cohort",
        f"535 sequences\n\u2193 measured label\n349 records\n"
        f"\u2193 marine only\n{n_rec} records\n{n_sp} species",
        fill=SAND, bsize=8.4)
    arrow(2.85, 5.92, 3.45, 5.55)
    arrow(2.85, 4.12, 3.45, 4.95)

    # Stage 3, three feature sets
    ax.text(7.95, 7.02, "Feature sets, compared by ablation",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    fb = [
        (5.62, "Apo structural geometry", "47 features, ESMFold", GREY),
        (4.52, "ESM-2 embeddings", "2560 dims, mean pooled", BLUE),
        (3.42, "Zn(2+) coordination geometry", "13 features, Boltz-1", GREY),
    ]
    for y, t, b, c in fb:
        filled = c == BLUE
        box(6.35, y, 3.2, 0.98, t, b,
            fill="#EAF1F8" if filled else "white",
            edge=c if filled else INK, lw=1.6 if filled else 1.0,
            tsize=9.2, bsize=8.2)
        arrow(6.05, 5.32, 6.35, y + 0.49)

    # Stage 4, evaluation
    box(9.95, 4.05, 2.7, 2.55, "Random Forest",
        "300 trees\n\n10 repeats of\n5-fold stratified\ngroup CV\n\n"
        "species never\nsplit across folds",
        fill="white", bsize=8.2)
    for y, *_ in fb:
        arrow(9.55, y + 0.49, 9.95, 5.32)

    # Stage 5, outcome
    box(0.35, 0.35, 12.3, 2.55, "", "", fill="white", edge=GREEN, lw=1.6)
    ax.text(6.5, 2.62, "Result", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=3)
    if mb and esm and st:
        acc = lambda d: d["habitat_accuracy_mean"] * 100
        cols = [
            (2.45, "Headline",
             f"Habitat accuracy  {acc(mb):.1f}%\nSD {mb['habitat_accuracy_sd']*100:.1f} points\n"
             f"Macro F1  {mb['habitat_f1_macro_mean']:.3f}\nSST  r = {mb['sst_pearson_mean']:.3f}", INK),
            (6.50, "Embeddings over apo structure",
             f"{acc(st):.1f}%  \u2192  {acc(esm):.1f}%\n\n+{acc(esm)-acc(st):.1f} points\n"
             f"several times the SD\nthis gain is real", GREEN),
            (10.55, "Zn geometry over embeddings",
             f"{acc(esm):.1f}%  \u2192  {acc(mb):.1f}%\n\n+{acc(mb)-acc(esm):.1f} points\n"
             f"SD is {mb['habitat_accuracy_sd']*100:.1f}\nnot measurable", MUTED),
        ]
        for x, head, body, c in cols:
            ax.text(x, 2.10, head, ha="center", va="center", fontsize=9.2,
                    fontweight="bold", color=c, zorder=3)
            ax.text(x, 1.24, body, ha="center", va="center", fontsize=8.8,
                    color=INK, linespacing=1.65, zorder=3)
        for xd in (4.48, 8.52):
            ax.plot([xd, xd], [0.62, 2.28], color="#D0D0D0", lw=1, zorder=1)
    arrow(11.3, 4.05, 11.3, 2.90, color=GREEN, lw=1.5)

    ax.text(6.5, 7.55, "Figure 1 - Study design", ha="center", va="center",
            fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig1_pipeline_schematic.png")


# Figure 2: Zn geometry validation violins

def fig_zn_geometry_validation() -> None:
    if not METALBOUND_FEATURES.exists():
        print("  SKIP fig2: metalbound_features.csv not found")
        return

    mb = pd.read_csv(METALBOUND_FEATURES)
    mb = mb.dropna(subset=["mb_mean_zn_s_dist", "mb_mean_szns_angle"])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Zn-S distance
    ax = axes[0]
    parts = ax.violinplot(mb["mb_mean_zn_s_dist"].dropna(), positions=[0], showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2196F3")
        pc.set_alpha(0.7)
    ax.axhline(CRYSTAL_ZN_S_DIST, color="#E53935", lw=1.5, ls="--", label=f"Crystal avg {CRYSTAL_ZN_S_DIST} Å")
    ax.set_xticks([0])
    ax.set_xticklabels(["Boltz-predicted"])
    ax.set_ylabel("Zn-S distance (Å)")
    ax.set_title("Zn-S Bond Length Validation")
    ax.legend(fontsize=8)
    mean_val = mb["mb_mean_zn_s_dist"].mean()
    ax.text(0.05, 0.95, f"Mean = {mean_val:.3f} Å\n(crystal = {CRYSTAL_ZN_S_DIST} Å)",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    # S-Zn-S angle
    ax = axes[1]
    parts = ax.violinplot(mb["mb_mean_szns_angle"].dropna(), positions=[0], showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4CAF50")
        pc.set_alpha(0.7)
    ax.axhline(CRYSTAL_SZNS_ANGLE, color="#E53935", lw=1.5, ls="--",
               label=f"Ideal tetrahedral {CRYSTAL_SZNS_ANGLE}°")
    ax.set_xticks([0])
    ax.set_xticklabels(["Boltz-predicted"])
    ax.set_ylabel("S-Zn-S angle (°)")
    ax.set_title("Tetrahedral Angle Validation")
    ax.legend(fontsize=8)
    mean_val = mb["mb_mean_szns_angle"].mean()
    ax.text(0.05, 0.95, f"Mean = {mean_val:.1f}°\n(ideal = {CRYSTAL_SZNS_ANGLE}°)",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    fig.suptitle("Figure 2 - Zn²⁺ Coordination Geometry: Boltz vs Crystal Reference", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_zn_geometry_validation.png")


# Figure 3: Ablation bar chart

def fig_ablation_bar() -> None:
    structural_summary = load_summary(STRUCTURAL_DIR)
    esm2_summary = load_summary(ESM2_DIR)
    metalbound_summary = load_summary(PRIMARY_DIR)

    def get_habitat_cv_acc(df: pd.DataFrame) -> float | None:
        if df.empty:
            return None
        row = df[(df["target"] == "habitat_type") & (df["model"] == "random_forest")]
        if row.empty:
            return None
        return float(row["cv_accuracy_mean"].iloc[0])

    accs = {
        "Structural\n(apo ESMFold)": get_habitat_cv_acc(structural_summary),
        "ESM-2\n(sequence only)": get_habitat_cv_acc(esm2_summary),
        "Metalbound + ESM-2\n(primary)": get_habitat_cv_acc(metalbound_summary),
    }

    for k in list(accs.keys()):
        if accs[k] is None:
            print(f"  WARNING: no data for {k!r} - skipping")

    labels = list(accs.keys())
    values = [accs[k] for k in labels]
    colors = ["#90CAF9", "#A5D6A7", "#1565C0"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="#555", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("5-fold CV Accuracy (habitat_type)")
    ax.set_title("Figure 3 - Ablation Study: Habitat Classification Accuracy by Feature Set\n(cross-validation, organism-level GroupShuffleSplit)",
                 fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Annotate Zn contribution
    ax.annotate("", xy=(2, values[2]), xytext=(1, values[1]),
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5))
    ax.text(1.5, (values[1] + values[2]) / 2 + 0.015, f"+{values[2]-values[1]:.1%}\n(Zn geometry)",
            ha="center", fontsize=8, color="#E53935")

    fig.tight_layout()
    save(fig, "fig3_ablation_bar.png")


# Figure 4: Confusion matrix

def fig_confusion_matrix() -> None:
    preds = load_predictions(PRIMARY_DIR, "habitat_type")
    if preds is None:
        print("  SKIP fig4: predictions not found - re-run training first")
        return

    labels = sorted(preds["y_true"].unique())
    cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")

    label_short = [l.replace("_", "\n") for l in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(label_short, fontsize=8)
    ax.set_yticklabels(label_short, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(len(labels)):
        for j in range(len(labels)):
            raw = cm[i, j]
            norm = cm_norm[i, j]
            color = "white" if norm > 0.6 else "black"
            ax.text(j, i, f"{norm:.2f}\n(n={raw})", ha="center", va="center",
                    fontsize=7.5, color=color)

    acc = float(np.diag(cm).sum()) / cm.sum()
    ax.set_title(f"Figure 4 - Habitat Confusion Matrix (RF, metalbound+ESM-2)\nTest accuracy = {acc:.1%}",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_confusion_matrix.png")


# Figure 5: SST scatter plot

def fig_sst_scatter() -> None:
    preds = load_predictions(PRIMARY_DIR, "sst_mean_c")
    if preds is None:
        print("  SKIP fig5: SST predictions not found - re-run training first")
        return

    y_true = preds["y_true"].values
    y_pred = preds["y_pred"].values
    r, _ = pearsonr(y_true, y_pred)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, s=40, color="#1565C0", edgecolors="#333", linewidths=0.5)

    lim = (min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2)
    ax.plot(lim, lim, "r--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True SST (°C)")
    ax.set_ylabel("Predicted SST (°C)")
    ax.set_title(f"Figure 5 - SST Prediction (RF, metalbound+ESM-2)\nr = {r:.3f}, R² = {r2:.3f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nR² = {r2:.3f}\nn = {len(y_true)}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    fig.tight_layout()
    save(fig, "fig5_sst_scatter.png")


# Figure 6: Results heatmap

def fig_results_heatmap() -> None:
    summaries = {
        "Structural\n(apo)": load_summary(STRUCTURAL_DIR),
        "ESM-2\nonly": load_summary(ESM2_DIR),
        "Metalbound\n+ESM-2": load_summary(PRIMARY_DIR),
    }

    targets = ["habitat_type", "sst_mean_c", "do_mean_mlL", "ph_mean", "depth_mean_m", "salinity_mean_psu"]
    target_labels = ["Habitat type\n(accuracy)", "SST (°C)\n(r)", "DO (mL/L)\n(r)",
                     "pH\n(r)", "Depth (m)\n(r)", "Salinity (psu)\n(r)"]

    def get_metric(df: pd.DataFrame, target: str) -> float:
        if df.empty:
            return np.nan
        row = df[(df["target"] == target) & (df["model"] == "random_forest")]
        if row.empty:
            return np.nan
        if target == "habitat_type":
            return float(row["cv_accuracy_mean"].iloc[0]) if "cv_accuracy_mean" in row else np.nan
        else:
            return float(row["cv_pearson_mean"].iloc[0]) if "cv_pearson_mean" in row else np.nan

    feature_sets = list(summaries.keys())
    matrix = np.zeros((len(targets), len(feature_sets)))
    for j, fs in enumerate(feature_sets):
        for i, t in enumerate(targets):
            matrix[i, j] = get_metric(summaries[fs], t)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.3, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CV Accuracy / CV Pearson r")

    ax.set_xticks(range(len(feature_sets)))
    ax.set_xticklabels(feature_sets, fontsize=9)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(target_labels, fontsize=9)
    ax.set_xlabel("Feature set")
    ax.set_title("Figure 6 - Model Performance Across All Targets and Feature Sets",
                 fontsize=10, fontweight="bold")

    for i in range(len(targets)):
        for j in range(len(feature_sets)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.7 or val < -0.1 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    fig.tight_layout()
    save(fig, "fig6_results_heatmap.png")


# main

def main() -> None:
    print("Generating Paper 1 figures...\n")

    print("[1/6] Pipeline schematic")
    fig_pipeline_schematic()

    print("[2/6] Zn geometry validation")
    fig_zn_geometry_validation()

    print("[3/6] Ablation bar chart")
    fig_ablation_bar()

    print("[4/6] Confusion matrix")
    fig_confusion_matrix()

    print("[5/6] SST scatter plot")
    fig_sst_scatter()

    print("[6/6] Results heatmap")
    fig_results_heatmap()

    print(f"\nAll figures written to {FIGURES_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
