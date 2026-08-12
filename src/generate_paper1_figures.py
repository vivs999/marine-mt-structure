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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.5, 1.2, "MT Sequences\n(535 seqs,\n272 species)", "#E3F2FD"),
        (2.5, 1.2, "Structure\nPrediction\n(ESMFold/Boltz)", "#E8F5E9"),
        (4.5, 1.2, "Feature\nExtraction\n(47 struct +\n2560 ESM-2 +\n13 Zn geometry)", "#FFF3E0"),
        (6.8, 1.2, "Random Forest\n+ Cross-validation\n(GroupShuffleSplit)", "#F3E5F5"),
        (9.0, 1.2, "Habitat /\nThermal\nPrediction", "#FCE4EC"),
    ]

    for (x, y, label, color) in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - 0.85, y - 0.7), 1.7, 1.4,
            boxstyle="round,pad=0.1",
            linewidth=1.2, edgecolor="#555",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, wrap=True)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 0.85
        x1 = boxes[i + 1][0] - 0.85
        y = 1.2
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.text(5.0, 0.15, "Zn²⁺ ligand geometry from Boltz-predicted metal-bound structures (novel contribution)",
            ha="center", va="center", fontsize=8, style="italic", color="#555")

    ax.set_title("Figure 1 - ML Pipeline Overview", fontsize=11, fontweight="bold", pad=8)
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
