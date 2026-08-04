#!/usr/bin/env python3
"""Compare fresh-case, history-borrowed, and full-data Beta posteriors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-history-borrowing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from history_borrowing.algorithm_2.bayesian_pseudo_posterior import (
    ModelOutcomes,
    beta_credible_interval,
    collapse_similarity_matrix,
    construct_history_prior,
    infer_replicate_map,
    load_full_model_outcomes,
    load_similarity_matrix,
    version_bundle_slices,
)


COLORS = {
    "fresh": "#8A94A6",
    "borrowed": "#2979FF",
    "full": "#F2A900",
}


def beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Evaluate a Beta density without requiring scipy."""
    log_normalizer = (
        math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    )
    log_density = (
        (alpha - 1.0) * np.log(x)
        + (beta - 1.0) * np.log1p(-x)
        - log_normalizer
    )
    return np.exp(log_density)


def posterior_summary(
    alpha: float, beta: float, credible_level: float
) -> dict[str, float]:
    lower, upper = beta_credible_interval(alpha, beta, credible_level)
    return {
        "alpha": alpha,
        "beta": beta,
        "mean": alpha / (alpha + beta),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def select_fresh_outcomes(
    full: ModelOutcomes,
    model_index: int,
    model_count: int,
    fresh_n: int,
    strategy: str,
) -> tuple[ModelOutcomes, str]:
    if strategy == "version_bundle":
        start, stop = version_bundle_slices(full.n, model_count)[model_index]
        bucket_label = (
            f"version bundle {model_index + 1} "
            f"(cases {start + 1}-{stop}, n={stop - start})"
        )
    elif strategy == "assigned":
        start = model_index * fresh_n
        stop = start + fresh_n
        bucket = model_index + 1
        bucket_label = f"bucket{bucket} (cases {start + 1}-{stop})"
    elif strategy == "first":
        start = 0
        stop = fresh_n
        bucket_label = f"first {fresh_n} cases"
    else:
        raise ValueError(f"Unknown fresh-case strategy: {strategy}")
    rewards = full.rewards[start:stop]
    expected_n = stop - start
    if len(rewards) != expected_n:
        raise ValueError(
            f"{full.model} has {full.n} outcomes; cannot select {bucket_label}"
        )
    return ModelOutcomes(full.model, rewards), bucket_label


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_outcomes = load_full_model_outcomes(Path(args.data_dir))
    models = [outcome.model for outcome in full_outcomes]
    labels, raw_matrix = load_similarity_matrix(Path(args.similarity_file))
    replicate_map = infer_replicate_map(models, labels)
    similarities = collapse_similarity_matrix(models, replicate_map, raw_matrix)

    rows: list[dict[str, Any]] = []
    deployed: list[ModelOutcomes] = []
    resolved_strategy = args.fresh_strategy

    for model_index, full in enumerate(full_outcomes):
        fresh, bucket_label = select_fresh_outcomes(
            full,
            model_index,
            len(models),
            args.fresh_n,
            resolved_strategy,
        )
        successes = sum(fresh.rewards)
        fresh_posterior = posterior_summary(
            args.alpha0 + successes,
            args.beta0 + fresh.n - successes,
            args.credible_level,
        )

        prior_alpha, prior_beta, weights = construct_history_prior(
            fresh.model,
            deployed,
            similarities,
            args.lambda_,
            args.alpha0,
            args.beta0,
        )
        borrowed_posterior = posterior_summary(
            prior_alpha + successes,
            prior_beta + fresh.n - successes,
            args.credible_level,
        )

        full_successes = sum(full.rewards)
        full_posterior = posterior_summary(
            args.alpha0 + full_successes,
            args.beta0 + full.n - full_successes,
            args.credible_level,
        )
        effective_history_n = sum(
            float(weight["borrowed_alpha_evidence"])
            + float(weight["borrowed_beta_evidence"])
            for weight in weights
        )
        rows.append(
            {
                "model": fresh.model,
                "deployment_step": model_index + 1,
                "history_models": " | ".join(item.model for item in deployed),
                "fresh_case_selection": bucket_label,
                "fresh_n": fresh.n,
                "fresh_successes": successes,
                "fresh_alpha": fresh_posterior["alpha"],
                "fresh_beta": fresh_posterior["beta"],
                "fresh_mean": fresh_posterior["mean"],
                "fresh_ci_lower": fresh_posterior["ci_lower"],
                "fresh_ci_upper": fresh_posterior["ci_upper"],
                "effective_history_n": effective_history_n,
                "borrowed_prior_alpha": prior_alpha,
                "borrowed_prior_beta": prior_beta,
                "borrowed_alpha": borrowed_posterior["alpha"],
                "borrowed_beta": borrowed_posterior["beta"],
                "borrowed_mean": borrowed_posterior["mean"],
                "borrowed_ci_lower": borrowed_posterior["ci_lower"],
                "borrowed_ci_upper": borrowed_posterior["ci_upper"],
                "full_n": full.n,
                "full_successes": full_successes,
                "full_alpha": full_posterior["alpha"],
                "full_beta": full_posterior["beta"],
                "full_mean": full_posterior["mean"],
                "full_ci_lower": full_posterior["ci_lower"],
                "full_ci_upper": full_posterior["ci_upper"],
            }
        )
        deployed.append(fresh)

    metadata = {
        "data_dir": args.data_dir,
        "similarity_file": args.similarity_file,
        "model_order": models,
        "bundle_sizes": [
            stop - start
            for start, stop in version_bundle_slices(
                full_outcomes[0].n, len(full_outcomes)
            )
        ]
        if resolved_strategy == "version_bundle"
        else [args.fresh_n] * len(full_outcomes),
        "fresh_strategy": resolved_strategy,
        "lambda": args.lambda_,
        "alpha0": args.alpha0,
        "beta0": args.beta0,
        "credible_level": args.credible_level,
        "interpretation": (
            "Fresh and history-borrowed posteriors use the same fresh outcomes. "
            "History borrowing adds unnormalised, similarity-weighted evidence "
            "from previously deployed models. Full posteriors use all available "
            "outcomes for each model."
        ),
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_interval(row: dict[str, Any], prefix: str) -> str:
    return f"({row[f'{prefix}_ci_lower']:.3f}, {row[f'{prefix}_ci_upper']:.3f})"


def write_markdown(path: Path, rows: list[dict[str, Any]], level: int) -> None:
    full_n_values = sorted({int(row["full_n"]) for row in rows})
    full_label = (
        f"Full {full_n_values[0]}-case posterior"
        if len(full_n_values) == 1
        else "Full-data posterior"
    )
    lines = [
        "# Algorithm 2 posterior distribution comparison",
        "",
        (
            f"| Model | Bundle n | Version-bundle 95% CI | "
            f"History-borrowed 95% CI | {full_label} 95% CI |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {int(row['fresh_n'])} | "
            f"{format_interval(row, 'fresh')} | "
            f"{format_interval(row, 'borrowed')} | "
            f"{format_interval(row, 'full')} |"
        )
    lines.extend(
        [
            "",
            f"Intervals are equal-tailed {level}% credible intervals.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_distributions(
    path: Path, rows: list[dict[str, Any]], dataset_name: str
) -> None:
    columns = 2
    row_count = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(12, 4.2 * row_count),
        squeeze=False,
        sharex=True,
    )
    x = np.linspace(0.001, 0.999, 1000)
    for axis, row in zip(axes.flat, rows):
        series = [
            (
                f"Version bundle ({int(row['fresh_n'])} cases)",
                "fresh",
                row["fresh_alpha"],
                row["fresh_beta"],
            ),
            (
                "History borrowed",
                "borrowed",
                row["borrowed_alpha"],
                row["borrowed_beta"],
            ),
            (
                f"Full {int(row['full_n'])} cases",
                "full",
                row["full_alpha"],
                row["full_beta"],
            ),
        ]
        for label, key, alpha, beta in series:
            density = beta_pdf(x, float(alpha), float(beta))
            axis.plot(x, density, color=COLORS[key], linewidth=2.2, label=label)
            axis.fill_between(x, density, color=COLORS[key], alpha=0.08)
        axis.set_title(str(row["model"]), fontsize=13, fontweight="bold")
        axis.set_xlabel("Diagnostic accuracy")
        axis.set_ylabel("Posterior density")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
    for axis in axes.flat[len(rows) :]:
        axis.axis("off")
    fig.suptitle(
        f"{dataset_name}: Algorithm 2 posterior distributions",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_interval_table(
    path: Path, rows: list[dict[str, Any]], dataset_name: str, level: int
) -> None:
    full_n_values = sorted({int(row["full_n"]) for row in rows})
    full_n_label = (
        str(full_n_values[0]) if len(full_n_values) == 1 else "all available"
    )
    headers = [
        "Model",
        "Bundle\nn",
        f"Version bundle\n{level}% CI",
        f"History borrowing\n{level}% CI",
        f"Full {full_n_label}-case posterior\n{level}% CI",
    ]
    cells = [
        [
            str(row["model"]),
            str(int(row["fresh_n"])),
            format_interval(row, "fresh"),
            format_interval(row, "borrowed"),
            format_interval(row, "full"),
        ]
        for row in rows
    ]
    fig, axis = plt.subplots(figsize=(14.5, 1.05 + 0.82 * len(rows)))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.19, 0.08, 0.23, 0.25, 0.25],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D9DDE5")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#EEF3FA")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#FAFBFD")
    fig.suptitle(
        f"{dataset_name}: Algorithm 2 credible intervals",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--similarity_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fresh_n", type=int, default=25)
    parser.add_argument(
        "--fresh_strategy",
        choices=("version_bundle", "assigned", "first"),
        default="version_bundle",
    )
    parser.add_argument("--lambda", dest="lambda_", type=float, default=10.0)
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--credible_level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fresh_n <= 0:
        raise SystemExit("ERROR: fresh_n must be positive")
    rows, metadata = build_rows(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "distribution_comparison.csv", rows)
    level = round(args.credible_level * 100)
    write_markdown(output_dir / "distribution_comparison.md", rows, level)
    plot_distributions(
        output_dir / "posterior_distributions.png", rows, args.dataset_name
    )
    plot_interval_table(
        output_dir / "credible_interval_comparison.png",
        rows,
        args.dataset_name,
        level,
    )
    (output_dir / "distribution_comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote distribution comparison for {len(rows)} models to {output_dir}")


if __name__ == "__main__":
    main()
