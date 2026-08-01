from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
# ── Styling ───────────────────────────────────────────────────────────
DARK_BG = "#14161A"
RAISED_BG = "#1B1E24"
TEXT_COL = "#E8E6E1"
DIM_COL = "#8B909C"
ACCENT = "#C98A3E"
GRID_COL = "#2A2E36"
CONFIG_COLOURS = {
    "fixed_300": "#5B8DBE",
    "fixed_500": "#E0A652",
    "semantic":  "#6FA875",
}
def apply_dark_style(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(RAISED_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=10)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.5)
# ── Chart 1: Config Comparison Bar Chart ──────────────────────────────
def chart_config_comparison():
    path = RESULTS_DIR / "comparison_summary.csv"
    if not path.exists():
        print("  [SKIP] comparison_summary.csv not found")
        return
    df = pd.read_csv(path)
    configs = df["config_name"].tolist()
    metrics = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_style(ax, fig)
    x = np.arange(len(configs))
    w = 0.22
    colours = ["#5B8DBE", "#E0A652", "#6FA875"]
    for i, metric in enumerate(metrics):
        vals = df[metric].tolist()
        bars = ax.bar(x + i * w, vals, w, label=metric.capitalize(), color=colours[i], edgecolor="none")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT_COL)
    ax.set_xlabel("Chunking Configuration", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall & F1 by Chunking Configuration", fontsize=14, fontweight="bold")
    ax.set_xticks(x + w)
    ax.set_xticklabels(configs)
    ax.set_ylim(0, max(df[metrics].max()) + 0.1)
    ax.legend(facecolor=RAISED_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    out = FIGURES_DIR / "config_comparison_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")
# ── Chart 2: Per-Policy F1 Box Plot ───────────────────────────────────
def chart_per_policy_boxplot():
    configs = ["fixed_300", "fixed_500", "semantic"]
    data = {}
    for c in configs:
        path = RESULTS_DIR / f"{c}_per_policy.csv"
        if path.exists():
            df = pd.read_csv(path)
            data[c] = df["f1"].tolist()
    if not data:
        print("  [SKIP] no per_policy CSVs found")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    apply_dark_style(ax, fig)
    bp = ax.boxplot(
        [data[c] for c in data],
        tick_labels=list(data.keys()),
        patch_artist=True,
        medianprops=dict(color=ACCENT, linewidth=2),
        whiskerprops=dict(color=DIM_COL),
        capprops=dict(color=DIM_COL),
        flierprops=dict(marker="o", markerfacecolor=DIM_COL, markersize=3, alpha=0.5),
    )
    for patch, colour in zip(bp["boxes"], [CONFIG_COLOURS.get(c, ACCENT) for c in data]):
        patch.set_facecolor(colour + "40")  # semi-transparent fill
        patch.set_edgecolor(colour)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Policy F1 Distribution by Chunking Configuration", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    out = FIGURES_DIR / "per_policy_f1_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")
# ── Chart 3: Per-Category Heatmap ─────────────────────────────────────
def chart_category_heatmap():
    path = RESULTS_DIR / "category_analysis.csv"
    if not path.exists():
        print("  [SKIP] category_analysis.csv not found")
        return
    df = pd.read_csv(path)
    pivot = df.pivot_table(index="category", columns="config", values="f1", aggfunc="mean")
    pivot = pivot.fillna(0)
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.45)))
    apply_dark_style(ax, fig)
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            colour = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=colour)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=TEXT_COL)
    cbar.set_label("F1 Score", color=TEXT_COL, fontsize=11)
    ax.set_title("F1 Score by Privacy Category and Chunking Config (Pilot-15)", fontsize=13, fontweight="bold")
    out = FIGURES_DIR / "category_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")
# ── Chart 4: Threshold Sensitivity ────────────────────────────────────
def chart_threshold_sensitivity():
    path = Path("experiments/threshold_sweep/results/all_configs_summary.csv")
    if not path.exists():
        print("  [SKIP] threshold_sweep all_configs_summary.csv not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_style(ax, fig)
    for config, colour in CONFIG_COLOURS.items():
        cfg_data = df[df["config"] == config].sort_values("threshold")
        if cfg_data.empty:
            continue
        ax.plot(cfg_data["threshold"], cfg_data["f1"], marker="o", color=colour,
                linewidth=2, markersize=6, label=config)
        ax.plot(cfg_data["threshold"], cfg_data["precision"], linestyle="--",
                color=colour, linewidth=1, alpha=0.6)
        ax.plot(cfg_data["threshold"], cfg_data["recall"], linestyle=":",
                color=colour, linewidth=1, alpha=0.6)
    ax.set_xlabel("Cosine Similarity Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Sensitivity: F1 (solid), Precision (dashed), Recall (dotted)",
                 fontsize=13, fontweight="bold")
    ax.legend(facecolor=RAISED_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=10)
    ax.set_xlim(-0.02, 0.42)
    ax.set_ylim(0, max(df["precision"].max(), 0.8) + 0.05)
    out = FIGURES_DIR / "threshold_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")
# ── Chart 5: Error Breakdown Stacked Bar ──────────────────────────────
def chart_error_breakdown():
    path = RESULTS_DIR / "error_analysis.csv"
    if not path.exists():
        print("  [SKIP] error_analysis.csv not found")
        return
    df = pd.read_csv(path)
    pivot = df.groupby(["config", "error_category"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_dark_style(ax, fig)
    cats = pivot.columns.tolist()
    cat_colours = {
        "perfect_match": "#6FA875",
        "missed_risk_only": "#E05252",
        "irrelevant_retrieval_only": "#E0A652",
        "mixed_errors": "#5B8DBE",
        "formatting_failure": "#9B59B6",
        "llm_timeout": "#8B909C",
        "no_evidence_retrieved": "#C15B4A",
        "no_predictions_no_gt": "#3A3F4B",
        "other": "#555555",
    }
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    for cat in cats:
        vals = pivot[cat].values.astype(float)
        colour = cat_colours.get(cat, "#888888")
        ax.bar(x, vals, bottom=bottom, label=cat.replace("_", " ").title(),
               color=colour, edgecolor="none", width=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("Chunking Configuration", fontsize=12)
    ax.set_ylabel("Number of Policies", fontsize=12)
    ax.set_title("Error Category Breakdown by Configuration", fontsize=14, fontweight="bold")
    ax.legend(facecolor=RAISED_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL,
              fontsize=8, loc="upper right", ncol=2)
    out = FIGURES_DIR / "error_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")
# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("Generating charts...\n")
    chart_config_comparison()
    chart_per_policy_boxplot()
    chart_category_heatmap()
    chart_threshold_sensitivity()
    chart_error_breakdown()
    print(f"\nAll charts saved to {FIGURES_DIR}/")
if __name__ == "__main__":
    main()