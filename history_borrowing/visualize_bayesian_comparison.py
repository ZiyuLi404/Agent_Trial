#!/usr/bin/env python3
"""
Compare MAE to ground truth for:
  1. 25-case only estimates
  2. original history borrowing
  3. Bayesian history borrowing

Default inputs compare the embedding-diagnosis original borrowing outputs with
the full-grid Bayesian update outputs.

Usage:
    python history_borrowing/visualize_bayesian_comparison.py

    python history_borrowing/visualize_bayesian_comparison.py \
        --original_source embedding_diagnosis \
        --bayesian_csv history_borrowing/data/results/bayesian_update_embedding_diagnosis_full_grid/bayesian_update_all_orders.csv
"""

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ORIGINAL_FULL_CSV = (
    "history_borrowing/data/results/compare/borrow_params_similarity_comparison_full.csv"
)
DEFAULT_BAYESIAN_CSV = (
    "history_borrowing/data/results/bayesian_update_embedding_diagnosis_full_grid/"
    "bayesian_update_all_orders.csv"
)
DEFAULT_OUTPUT_DIR = (
    "history_borrowing/data/results/bayesian_update_embedding_diagnosis_full_grid"
)

COLORS = {
    "baseline": "#8a8f98",
    "original": "#2d6cdf",
    "bayesian": "#d94f45",
    "grid": "#d8dde6",
    "text": "#23272f",
}


def mean(values) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sort_perm_key(tag: str) -> tuple:
    parts = []
    for part in tag.split("_"):
        try:
            parts.append(int(part.replace("b", "")))
        except ValueError:
            parts.append(part)
    return tuple(parts)


def load_original(rows: list[dict], source: str) -> tuple[list[str], dict, dict, dict, dict]:
    src_rows = [r for r in rows if r.get("similarity_source") == source]
    if not src_rows:
        available = sorted({r.get("similarity_source", "") for r in rows})
        sys.exit(f"ERROR: no original rows for source='{source}'. Available: {available}")

    models = []
    for row in src_rows:
        model = row["model"]
        if model not in models:
            models.append(model)

    baseline_by_perm: dict[str, list[float]] = defaultdict(list)
    original_by_perm: dict[str, list[float]] = defaultdict(list)
    baseline_by_model: dict[str, list[float]] = defaultdict(list)
    original_by_model: dict[str, list[float]] = defaultdict(list)

    for row in src_rows:
        perm = row["permutation"]
        model = row["model"]
        baseline_by_perm[perm].append(float(row["abs_error_25"]))
        original_by_perm[perm].append(float(row["abs_error_borrowed"]))
        baseline_by_model[model].append(float(row["abs_error_25"]))
        original_by_model[model].append(float(row["abs_error_borrowed"]))

    return (
        models,
        {p: mean(v) for p, v in baseline_by_perm.items()},
        {p: mean(v) for p, v in original_by_perm.items()},
        {m: mean(v) for m, v in baseline_by_model.items()},
        {m: mean(v) for m, v in original_by_model.items()},
    )


def bayesian_perm_from_order(order: str, selected_rows: list[dict], models: list[str]) -> str:
    bucket_by_model = {r["target_model"]: r["bucket_id"] for r in selected_rows}
    if not all(m in bucket_by_model for m in models):
        order_models = [m.strip() for m in order.split("->")]
        bucket_by_model = {model: f"bucket{i + 1}" for i, model in enumerate(order_models)}
    return "_".join(f"b{bucket_by_model[m].replace('bucket', '')}" for m in models)


def choose_bayesian_params(
    rows: list[dict],
    similarity_mode: str,
    alpha: Optional[float],
    lam: Optional[float],
) -> tuple[float, float]:
    mode_rows = [r for r in rows if r.get("similarity_mode") == similarity_mode]
    if not mode_rows:
        available = sorted({r.get("similarity_mode", "") for r in rows})
        sys.exit(f"ERROR: no Bayesian rows for similarity_mode='{similarity_mode}'. Available: {available}")

    if alpha is not None and lam is not None:
        return alpha, lam
    if alpha is not None or lam is not None:
        sys.exit("ERROR: pass both --bayesian_alpha and --bayesian_lambda, or neither.")

    errors_by_param: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in mode_rows:
        key = (float(row["alpha"]), float(row["lambda"]))
        errors_by_param[key].append(float(row["absolute_error"]))

    return min(
        errors_by_param,
        key=lambda key: (mean(errors_by_param[key]), key[0], key[1]),
    )


def load_bayesian(
    rows: list[dict],
    models: list[str],
    similarity_mode: str,
    alpha: float,
    lam: float,
) -> tuple[dict, dict, float]:
    selected = [
        r for r in rows
        if r.get("similarity_mode") == similarity_mode
        and float(r["alpha"]) == alpha
        and float(r["lambda"]) == lam
    ]
    if not selected:
        sys.exit(
            f"ERROR: no Bayesian rows for mode={similarity_mode}, "
            f"alpha={alpha}, lambda={lam}"
        )

    by_order: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_order[row["model_order"]].append(row)

    bayesian_by_perm: dict[str, list[float]] = defaultdict(list)
    bayesian_by_model: dict[str, list[float]] = defaultdict(list)

    for order, order_rows in by_order.items():
        perm = bayesian_perm_from_order(order, order_rows, models)
        for row in order_rows:
            err = float(row["absolute_error"])
            bayesian_by_perm[perm].append(err)
            bayesian_by_model[row["target_model"]].append(err)

    return (
        {p: mean(v) for p, v in bayesian_by_perm.items()},
        {m: mean(v) for m, v in bayesian_by_model.items()},
        mean([float(r["absolute_error"]) for r in selected]),
    )


def pct_change(before: float, after: float) -> float:
    return (before - after) / before * 100.0 if before > 0 else float("nan")


def annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["text"],
            fontweight="bold",
        )


def build_plot(
    output_png: Path,
    source: str,
    bayesian_mode: str,
    original_alpha: float,
    original_lambda: float,
    bayesian_alpha: float,
    bayesian_lambda: float,
    order_rows: list[dict],
    model_rows: list[dict],
    summary_rows: list[dict],
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    methods = ["25-case only", "Original borrowing", "Bayesian borrowing"]
    global_mae = [float(r["global_mae"]) for r in summary_rows]

    perms = [r["permutation"] for r in order_rows]
    x = np.arange(len(perms))

    fig = plt.figure(figsize=(18, 11), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.22], width_ratios=[0.46, 0.54])

    ax_global = fig.add_subplot(gs[0, 0])
    bars = ax_global.bar(
        np.arange(3),
        global_mae,
        color=[COLORS["baseline"], COLORS["original"], COLORS["bayesian"]],
        width=0.58,
    )
    annotate_bars(ax_global, bars)
    ax_global.set_xticks(np.arange(3))
    ax_global.set_xticklabels(methods, rotation=0, fontsize=9)
    ax_global.set_ylabel("Mean absolute error")
    ax_global.set_title("Global MAE to Ground Truth", fontweight="bold")
    ax_global.set_ylim(0, max(global_mae) * 1.35)
    ax_global.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax_global.spines[["top", "right"]].set_visible(False)

    base = global_mae[0]
    note = (
        f"Source: {source}\n"
        f"Original history borrowing: alpha={original_alpha:g}, lambda={original_lambda:g}\n"
        f"Bayesian borrowing: mode={bayesian_mode}, alpha={bayesian_alpha:g}, lambda={bayesian_lambda:g}\n\n"
        f"Original vs 25-case: {pct_change(base, global_mae[1]):+.1f}% MAE change\n"
        f"Bayesian vs 25-case: {pct_change(base, global_mae[2]):+.1f}% MAE change\n"
        f"Bayesian vs original: {pct_change(global_mae[1], global_mae[2]):+.1f}% MAE change"
    )
    ax_global.text(
        0.02,
        0.96,
        note,
        transform=ax_global.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "#f7f9fc", "edgecolor": "#d2d8e2", "boxstyle": "round,pad=0.5"},
    )

    ax_order = fig.add_subplot(gs[0, 1])
    ax_order.plot(
        x,
        [float(r["mae_25_case_only"]) for r in order_rows],
        marker="o",
        markersize=3.2,
        linewidth=1.4,
        color=COLORS["baseline"],
        label="25-case only",
    )
    ax_order.plot(
        x,
        [float(r["mae_original_borrowing"]) for r in order_rows],
        marker="o",
        markersize=3.2,
        linewidth=1.4,
        color=COLORS["original"],
        label="Original borrowing",
    )
    ax_order.plot(
        x,
        [float(r["mae_bayesian_borrowing"]) for r in order_rows],
        marker="o",
        markersize=3.2,
        linewidth=1.4,
        color=COLORS["bayesian"],
        label="Bayesian borrowing",
    )
    ax_order.set_xticks(x)
    ax_order.set_xticklabels(perms, rotation=90, fontsize=7)
    ax_order.set_ylabel("MAE")
    ax_order.set_title("MAE Across All 24 Bucket/Update Orders", fontweight="bold")
    ax_order.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax_order.spines[["top", "right"]].set_visible(False)
    ax_order.legend(frameon=False, fontsize=8)

    ax_model = fig.add_subplot(gs[1, :])
    models = [r["model"] for r in model_rows]
    idx = np.arange(len(models))
    width = 0.24
    bars1 = ax_model.bar(
        idx - width,
        [float(r["mae_25_case_only"]) for r in model_rows],
        width,
        color=COLORS["baseline"],
        label="25-case only",
    )
    bars2 = ax_model.bar(
        idx,
        [float(r["mae_original_borrowing"]) for r in model_rows],
        width,
        color=COLORS["original"],
        label="Original borrowing",
    )
    bars3 = ax_model.bar(
        idx + width,
        [float(r["mae_bayesian_borrowing"]) for r in model_rows],
        width,
        color=COLORS["bayesian"],
        label="Bayesian borrowing",
    )
    annotate_bars(ax_model, bars1)
    annotate_bars(ax_model, bars2)
    annotate_bars(ax_model, bars3)
    ax_model.set_xticks(idx)
    ax_model.set_xticklabels(models, fontsize=10)
    ax_model.set_ylabel("Mean absolute error")
    ax_model.set_title("Per-Model MAE to Ground Truth", fontweight="bold")
    ax_model.set_ylim(
        0,
        max(
            max(float(r["mae_25_case_only"]) for r in model_rows),
            max(float(r["mae_original_borrowing"]) for r in model_rows),
            max(float(r["mae_bayesian_borrowing"]) for r in model_rows),
        ) * 1.35,
    )
    ax_model.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax_model.spines[["top", "right"]].set_visible(False)
    ax_model.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle(
        "MAE to Ground Truth: 25-Case Baseline vs Original and Bayesian Borrowing",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize MAE comparison between baseline, original borrowing, and Bayesian borrowing."
    )
    parser.add_argument("--original_full_csv", default=DEFAULT_ORIGINAL_FULL_CSV)
    parser.add_argument("--original_source", default="embedding_diagnosis")
    parser.add_argument("--bayesian_csv", default=DEFAULT_BAYESIAN_CSV)
    parser.add_argument("--bayesian_mode", default="embedding")
    parser.add_argument("--bayesian_alpha", type=float, default=None)
    parser.add_argument("--bayesian_lambda", type=float, default=None)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_prefix", default="mae_comparison")
    args = parser.parse_args()

    original_rows = read_csv(Path(args.original_full_csv))
    bayesian_rows = read_csv(Path(args.bayesian_csv))

    (
        models,
        baseline_by_perm,
        original_by_perm,
        baseline_by_model,
        original_by_model,
    ) = load_original(original_rows, args.original_source)

    bayes_alpha, bayes_lam = choose_bayesian_params(
        bayesian_rows,
        args.bayesian_mode,
        args.bayesian_alpha,
        args.bayesian_lambda,
    )
    bayesian_by_perm, bayesian_by_model, bayesian_global = load_bayesian(
        bayesian_rows,
        models,
        args.bayesian_mode,
        bayes_alpha,
        bayes_lam,
    )

    missing = sorted(set(baseline_by_perm) - set(bayesian_by_perm), key=sort_perm_key)
    if missing:
        sys.exit(f"ERROR: Bayesian file is missing permutation(s): {missing}")

    original_source_rows = [r for r in original_rows if r.get("similarity_source") == args.original_source]
    original_alpha = float(original_source_rows[0]["alpha"])
    original_lam = float(original_source_rows[0]["lambda"])

    permutations = sorted(baseline_by_perm, key=sort_perm_key)
    order_out = []
    for perm in permutations:
        b = baseline_by_perm[perm]
        o = original_by_perm[perm]
        y = bayesian_by_perm[perm]
        order_out.append({
            "permutation": perm,
            "mae_25_case_only": round(b, 6),
            "mae_original_borrowing": round(o, 6),
            "mae_bayesian_borrowing": round(y, 6),
            "bayesian_minus_original": round(y - o, 6),
            "bayesian_improvement_vs_25_pct": round(pct_change(b, y), 6),
        })

    model_out = []
    for model in models:
        b = baseline_by_model[model]
        o = original_by_model[model]
        y = bayesian_by_model[model]
        model_out.append({
            "model": model,
            "mae_25_case_only": round(b, 6),
            "mae_original_borrowing": round(o, 6),
            "mae_bayesian_borrowing": round(y, 6),
            "bayesian_minus_original": round(y - o, 6),
            "bayesian_improvement_vs_25_pct": round(pct_change(b, y), 6),
        })

    global_25 = mean(baseline_by_perm.values())
    global_original = mean(original_by_perm.values())
    summary_out = [
        {
            "method": "25-case only",
            "global_mae": round(global_25, 6),
            "source": args.original_source,
            "alpha": "",
            "lambda": "",
        },
        {
            "method": "original history borrowing",
            "global_mae": round(global_original, 6),
            "source": args.original_source,
            "alpha": original_alpha,
            "lambda": original_lam,
        },
        {
            "method": "bayesian borrowing",
            "global_mae": round(bayesian_global, 6),
            "source": args.bayesian_mode,
            "alpha": bayes_alpha,
            "lambda": bayes_lam,
        },
    ]

    output_dir = Path(args.output_dir)
    summary_csv = output_dir / f"{args.output_prefix}_summary.csv"
    order_csv = output_dir / f"{args.output_prefix}_by_order.csv"
    model_csv = output_dir / f"{args.output_prefix}_by_model.csv"
    output_png = output_dir / f"{args.output_prefix}.png"

    write_csv(summary_csv, summary_out)
    write_csv(order_csv, order_out)
    write_csv(model_csv, model_out)
    build_plot(
        output_png=output_png,
        source=args.original_source,
        bayesian_mode=args.bayesian_mode,
        original_alpha=original_alpha,
        original_lambda=original_lam,
        bayesian_alpha=bayes_alpha,
        bayesian_lambda=bayes_lam,
        order_rows=order_out,
        model_rows=model_out,
        summary_rows=summary_out,
    )

    print("MAE comparison complete.")
    print(f"  25-case only MAE          : {global_25:.6f}")
    print(f"  Original borrowing MAE    : {global_original:.6f} (alpha={original_alpha:g}, lambda={original_lam:g})")
    print(f"  Bayesian borrowing MAE    : {bayesian_global:.6f} (alpha={bayes_alpha:g}, lambda={bayes_lam:g})")
    print(f"  PNG                       : {output_png}")
    print(f"  Summary CSV               : {summary_csv}")
    print(f"  By-order CSV              : {order_csv}")
    print(f"  By-model CSV              : {model_csv}")


if __name__ == "__main__":
    main()
