"""
Visualize global history-borrowing results for one similarity source.

Produces a dashboard that mirrors the reference layout:
  - Title bar with alpha / lambda
  - Top row: summary table | description box | overall MAE line chart
  - Bottom 2×2: per-model MAE line chart + bar chart panels

Usage:
    # Diagnosis (default)
    python history_borrowing/visualize_borrow_params.py

    # Conversation
    python history_borrowing/visualize_borrow_params.py --source conversation

    # Fingerprint conversation
    python history_borrowing/visualize_borrow_params.py --source fingerprint_conversation

    # Custom output path
    python history_borrowing/visualize_borrow_params.py \
        --source diagnosis \
        --output history_borrowing/data/results/diagnosis_visualization.png
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
GREEN     = "#2ca02c"
L_GREEN   = "#d4edda"
GRAY      = "#aaaaaa"
D_GRAY    = "#444444"
BLUE      = "#1a4a8a"
L_BLUE    = "#dce8f8"
OFF_WHITE = "#f9f9f9"

SOURCE_LABELS = {
    "diagnosis":    "Diagnosis-Only Similarity",
    "conversation": "Full-Conversation Similarity",
    "fingerprint_conversation": "Fingerprint Conversation Similarity",
    "embedding_diagnosis": "Embedding Diagnosis Similarity",
    "hybrid_0.7_embedding_diagnosis_0.3_fingerprint_conversation": "Hybrid Similarity (0.7/0.3)",
}

SOURCE_SHORT_LABELS = {
    "diagnosis": "diagnosis",
    "conversation": "conversation",
    "fingerprint_conversation": "fingerprint",
    "embedding_diagnosis": "embedding diagnosis",
    "hybrid_0.7_embedding_diagnosis_0.3_fingerprint_conversation": "hybrid 0.7/0.3",
}


def mean(values) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# Sub-plot drawing helpers
# ---------------------------------------------------------------------------

def draw_lineplot(ax, perms, y_before, y_after, *,
                  title, after_label,
                  title_fontsize=9, val_fontsize=5.5, show_legend=True):
    """Gray before-line vs green after-line across 24 permutations."""
    x = np.arange(len(perms))

    ax.plot(x, y_before, color=GRAY,  marker="o", ms=3.5, lw=1.2,
            label="MAE before (25-patient only)", zorder=3)
    ax.plot(x, y_after,  color=GREEN, marker="o", ms=3.5, lw=1.2,
            label=after_label, zorder=3)

    y_max = max(max(y_before), max(y_after))
    gap   = y_max * 0.06

    for xi, (yb, ya) in enumerate(zip(y_before, y_after)):
        ax.text(xi, yb + gap, f"{yb:.3f}",
                ha="center", va="bottom", fontsize=val_fontsize,
                color=D_GRAY, zorder=4)
        ax.text(xi, ya + gap, f"{ya:.3f}",
                ha="center", va="bottom", fontsize=val_fontsize,
                color=GREEN, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(perms, rotation=90, fontsize=4.5, ha="center")
    ax.set_ylabel("MAE", fontsize=7)
    ax.set_ylim(0, y_max * 1.6)
    ax.set_title(title, fontsize=title_fontsize, color=BLUE,
                 fontweight="bold", pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(axis="x", length=0)

    if show_legend:
        ax.legend(fontsize=5.5, loc="upper right", framealpha=0.85,
                  handlelength=1.2, borderpad=0.4, labelspacing=0.3)


def draw_barplot(ax, avg_before, avg_after, source_short):
    """Two-bar chart (Before / After) with improvement label below."""
    bars = ax.bar([0, 1], [avg_before, avg_after],
                  color=[GRAY, GREEN], width=0.5,
                  edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, [avg_before, avg_after]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    imp = (avg_before - avg_after) / avg_before * 100 if avg_before > 0 else 0.0
    color = GREEN if imp > 0 else "red"
    sign  = "+" if imp > 0 else ""
    ax.text(0.5, -0.32, f"Improvement: {sign}{imp:.1f}%",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=8, color=color, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Before\n(25-patient\nonly)", f"After\n({source_short})"],
                       fontsize=6)
    ax.set_title("MAE SUMMARY\n(ACROSS 24 ORDERS)",
                 fontsize=7.5, color=BLUE, fontweight="bold", pad=3)
    ax.set_ylim(0, max(avg_before, avg_after) * 1.60)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(axis="x", length=0)


def draw_summary_table(ax, source, alpha, lam, global_mae_25, global_mae_bor):
    """Styled overall-summary table."""
    ax.axis("off")
    ax.set_facecolor(L_BLUE)

    imp_abs = global_mae_25 - global_mae_bor
    imp_pct = imp_abs / global_mae_25 * 100 if global_mae_25 > 0 else 0.0

    ax.text(0.5, 0.97,
            "OVERALL MAE ACROSS ALL MODELS AND BUCKET ORDERS",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=BLUE)

    col_labels = [
        "Source", "alpha", "lambda",
        "MAE before\n(25-patient only)",
        "MAE after\n(history borrowing)",
        "Improvement\n(abs)",
        "Improvement\n(%)",
    ]
    cell_data = [[
        source, str(alpha), str(int(lam)),
        f"{global_mae_25:.4f}",
        f"{global_mae_bor:.4f}",
        f"{imp_abs:.4f}",
        f"{imp_pct:.1f}%",
    ]]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.80],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    n_cols = len(col_labels)
    for j in range(n_cols):
        # Header row (row index 0)
        hcell = tbl[(0, j)]
        hcell.set_facecolor(BLUE)
        hcell.set_text_props(color="white", fontweight="bold")
        hcell.set_edgecolor("white")
        hcell.set_height(0.45)

        # Data row (row index 1)
        dcell = tbl[(1, j)]
        if j >= 5:                          # Improvement columns → green tint
            dcell.set_facecolor(L_GREEN)
            dcell.set_text_props(color=GREEN, fontweight="bold")
        else:
            dcell.set_facecolor("white")
        dcell.set_edgecolor("#dddddd")
        dcell.set_height(0.45)


def draw_description(ax, source_label, alpha, lam, global_mae_25, global_mae_bor):
    """Description text box with big improvement percentage."""
    ax.axis("off")
    imp_pct = (global_mae_25 - global_mae_bor) / global_mae_25 * 100 if global_mae_25 > 0 else 0.0

    rect = mpatches.FancyBboxPatch(
        (0.03, 0.03), 0.94, 0.94,
        boxstyle="round,pad=0.01",
        transform=ax.transAxes,
        facecolor=OFF_WHITE, edgecolor="#cccccc", linewidth=1.2, zorder=0,
    )
    ax.add_patch(rect)

    ax.text(0.5, 0.76,
            f"Using {source_label}\nwith alpha = {alpha}\nand lambda = {int(lam)}\nreduces MAE by",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color="black", multialignment="center", linespacing=1.6, zorder=1)

    color = GREEN if imp_pct > 0 else "red"
    ax.text(0.5, 0.40, f"{imp_pct:.1f}%",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=22, color=color, fontweight="bold", zorder=1)

    ax.text(0.5, 0.16,
            "overall across all models\nand all 24 bucket orders.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color="black", multialignment="center", zorder=1)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(
    source, source_label, alpha, lam,
    global_mae_25, global_mae_bor,
    perms, perm_mae_25, perm_mae_bor,
    models, model_stats,
    output_path,
):
    fig = plt.figure(figsize=(22, 15), facecolor="white")
    source_short = SOURCE_SHORT_LABELS.get(source, source)

    fig.suptitle(
        f"{source_label}   (alpha = {alpha},  lambda = {int(lam)})",
        fontsize=18, fontweight="bold", color=BLUE, y=0.985,
    )

    # 3-row outer layout: tiny spacer | top panel | 2×2 model panels
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.01, 0.285, 0.645],
        hspace=0.30,
        top=0.945, bottom=0.03,
    )

    # ── Top row ─────────────────────────────────────────────────────────────
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        width_ratios=[0.40, 0.16, 0.44],
        wspace=0.05,
    )

    table_ax = fig.add_subplot(top_gs[0])
    table_ax.set_facecolor(L_BLUE)
    draw_summary_table(table_ax, source_short, alpha, lam, global_mae_25, global_mae_bor)

    desc_ax = fig.add_subplot(top_gs[1])
    draw_description(desc_ax, source_label, alpha, lam, global_mae_25, global_mae_bor)

    overall_ax = fig.add_subplot(top_gs[2])
    draw_lineplot(
        overall_ax, perms, perm_mae_25, perm_mae_bor,
        title="MAE BY BUCKET ORDER ACROSS ALL MODELS",
        after_label=f"MAE after ({source_short}, α={alpha}, λ={int(lam)})",
        title_fontsize=9.5, val_fontsize=5.5,
    )

    # ── Bottom 2 × 2 ────────────────────────────────────────────────────────
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[2],
        hspace=0.55, wspace=0.06,
    )

    for idx, model in enumerate(models):
        row, col = idx // 2, idx % 2
        panel_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=bottom_gs[row, col],
            width_ratios=[0.73, 0.27], wspace=0.14,
        )
        line_ax = fig.add_subplot(panel_gs[0])
        bar_ax  = fig.add_subplot(panel_gs[1])

        ms = model_stats[model]

        # Model name header above the sub-panel
        line_ax.annotate(
            f"{idx + 1}. {model}",
            xy=(0.0, 1.18), xycoords="axes fraction",
            fontsize=10.5, fontweight="bold", color=BLUE,
            ha="left", va="bottom", clip_on=False,
        )

        draw_lineplot(
            line_ax, perms,
            ms["mae_25_list"], ms["mae_bor_list"],
            title="MAE BY BUCKET ORDER",
            after_label=f"After ({source_short})",
            title_fontsize=8, val_fontsize=4.8,
        )
        draw_barplot(bar_ax, ms["avg_mae_25"], ms["avg_mae_bor"], source_short)

    # Shared legend note at the very bottom
    fig.text(
        0.5, 0.005,
        f"Notes:  MAE = Mean Absolute Error.  "
        f"Before = 25-patient only estimate vs 100-patient accuracy.  "
        f"After = History borrowing estimate ({source_label}, α={alpha}, λ={int(lam)}) vs 100-patient accuracy.",
        ha="center", va="bottom", fontsize=7, color=D_GRAY,
        style="italic",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize global borrowing results for a similarity source."
    )
    parser.add_argument(
        "--source", default="diagnosis",
        choices=sorted(SOURCE_LABELS),
        help="Which similarity source to visualise (default: diagnosis).",
    )
    parser.add_argument(
        "--full_csv",
        default="history_borrowing/data/results/borrow_params_full.csv",
        help="Long-form detail CSV from train_borrow_params.py.",
    )
    parser.add_argument(
        "--params_json",
        default="history_borrowing/data/results/borrow_params.json",
        help="Parameter JSON from train_borrow_params.py.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path (default: history_borrowing/<source>_visualization.png).",
    )
    args = parser.parse_args()

    source      = args.source
    output_path = args.output or f"history_borrowing/data/results/{source}_visualization.png"

    # ── Load data ────────────────────────────────────────────────────────────
    full_csv = Path(args.full_csv)
    if not full_csv.exists():
        sys.exit(f"ERROR: {full_csv} not found. Run train_borrow_params.py first.")

    with open(full_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    src_rows = [r for r in rows if r["similarity_source"] == source]
    if not src_rows:
        sys.exit(f"ERROR: no rows for source='{source}' in {full_csv}")

    params_path = Path(args.params_json)
    if not params_path.exists():
        sys.exit(f"ERROR: {params_path} not found. Run train_borrow_params.py first.")

    with open(params_path) as f:
        params = json.load(f)

    alpha          = params[f"best_alpha_{source}"]
    lam            = params[f"best_lambda_{source}"]
    global_mae_25  = params[f"global_mae_25_{source}"]
    global_mae_bor = params[f"global_mae_borrowed_{source}"]
    source_label   = SOURCE_LABELS[source]

    # ── Organise data ────────────────────────────────────────────────────────
    perms  = sorted({r["permutation"] for r in src_rows})
    models = list(dict.fromkeys(r["model"] for r in src_rows if r["permutation"] == perms[0]))

    perm_mae_25, perm_mae_bor = [], []
    for perm in perms:
        perm_rows = [r for r in src_rows if r["permutation"] == perm]
        perm_mae_25.append(mean(float(r["abs_error_25"]) for r in perm_rows))
        perm_mae_bor.append(mean(float(r["abs_error_borrowed"]) for r in perm_rows))

    model_stats: dict = {}
    for model in models:
        model_rows = [r for r in src_rows if r["model"] == model]
        m25, mbor = [], []
        for perm in perms:
            row = next((r for r in model_rows if r["permutation"] == perm), None)
            m25.append(float(row["abs_error_25"]) if row else np.nan)
            mbor.append(float(row["abs_error_borrowed"]) if row else np.nan)
        model_stats[model] = {
            "mae_25_list":  m25,
            "mae_bor_list": mbor,
            "avg_mae_25":   mean(m25),
            "avg_mae_bor":  mean(mbor),
        }

    print(f"\nBuilding {source_label} visualization …")
    print(f"  alpha={alpha}, lambda={int(lam)}")
    print(f"  Global MAE: {global_mae_25:.4f} → {global_mae_bor:.4f} "
          f"({(global_mae_25-global_mae_bor)/global_mae_25*100:.1f}% improvement)")
    for m in models:
        ms = model_stats[m]
        imp = (ms['avg_mae_25'] - ms['avg_mae_bor']) / ms['avg_mae_25'] * 100
        print(f"  {m:<25} {ms['avg_mae_25']:.4f} → {ms['avg_mae_bor']:.4f}  ({imp:+.1f}%)")
    print()

    build_figure(
        source, source_label, alpha, lam,
        global_mae_25, global_mae_bor,
        perms, perm_mae_25, perm_mae_bor,
        models, model_stats,
        output_path,
    )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Example commands
# ---------------------------------------------------------------------------
#
# Diagnosis (default):
# python history_borrowing/visualize_borrow_params.py
#
# Conversation:
# python history_borrowing/visualize_borrow_params.py --source conversation
#
# Fingerprint conversation:
# python history_borrowing/visualize_borrow_params.py --source fingerprint_conversation
#
# Custom output:
# python history_borrowing/visualize_borrow_params.py \
#     --source diagnosis \
#     --output history_borrowing/data/results/diagnosis_visualization.png
