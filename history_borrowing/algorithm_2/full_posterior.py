#!/usr/bin/env python3
"""Compute a full-data Beta posterior for every evaluated model/version.

The input is the same as the history-borrowing experiments: a directory with
one scored JSON file per model/version. Each file may contain a top-level list,
a recognized record-list key, or one scored record. Correctness may appear in
the flat or nested fields accepted by ``bayesian_pseudo_posterior.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

if __package__:
    from history_borrowing.algorithm_2.bayesian_pseudo_posterior import (
        ModelOutcomes,
        beta_credible_interval,
        load_full_model_outcomes,
    )
else:  # Support direct execution by file path.
    from bayesian_pseudo_posterior import (  # type: ignore[no-redef]
        ModelOutcomes,
        beta_credible_interval,
        load_full_model_outcomes,
    )


FIELDNAMES = [
    "model",
    "n_full",
    "accuracy_full",
    "alpha_full",
    "beta_full",
    "posterior_mean",
    "posterior_variance",
    "lower_95_ci",
    "upper_95_ci",
]


def compute_full_posterior(
    outcome: ModelOutcomes,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> dict[str, Any]:
    """Return the full-data Beta posterior summary for one model/version."""
    if alpha0 <= 0.0 or beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be positive")
    if outcome.n == 0:
        raise ValueError(f"Model '{outcome.model}' has no scored cases")

    n_full = outcome.n
    successes = math.fsum(outcome.rewards)
    failures = math.fsum(1.0 - reward for reward in outcome.rewards)
    accuracy_full = successes / n_full
    # These evidence sums are algebraically n_full * accuracy_full and
    # n_full * (1 - accuracy_full), with fewer floating-point artifacts.
    alpha_full = alpha0 + successes
    beta_full = beta0 + failures
    total = alpha_full + beta_full
    posterior_mean = alpha_full / total
    posterior_variance = alpha_full * beta_full / (total**2 * (total + 1.0))
    lower_95_ci, upper_95_ci = beta_credible_interval(
        alpha_full, beta_full, 0.95
    )

    return {
        "model": outcome.model,
        "n_full": n_full,
        "accuracy_full": accuracy_full,
        "alpha_full": alpha_full,
        "beta_full": beta_full,
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance,
        "lower_95_ci": lower_95_ci,
        "upper_95_ci": upper_95_ci,
    }


def build_full_posterior_rows(
    outcomes: list[ModelOutcomes],
    alpha0: float = 1.0,
    beta0: float = 1.0,
    expected_n: int | None = 100,
) -> list[dict[str, Any]]:
    """Compute summaries and optionally require a full case count per model."""
    if expected_n is not None and expected_n <= 0:
        raise ValueError("expected_n must be positive")

    rows = []
    for outcome in outcomes:
        if expected_n is not None and outcome.n != expected_n:
            raise ValueError(
                f"Model '{outcome.model}' has {outcome.n} valid scored cases; "
                f"expected {expected_n}"
            )
        rows.append(compute_full_posterior(outcome, alpha0, beta0))
    return rows


def write_full_posterior(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write full posterior summaries using the required output schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the full-data Beta posterior for each model/version."
    )
    parser.add_argument(
        "--data_dir",
        default="history_borrowing/data/groundtruth",
        help="Directory containing one scored JSON result file per model/version.",
    )
    parser.add_argument(
        "--output_csv",
        default="history_borrowing/data/results/algorithm_2/full_posterior.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument(
        "--expected_n",
        type=int,
        default=100,
        help="Required valid case count per model/version (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes = load_full_model_outcomes(Path(args.data_dir))
    rows = build_full_posterior_rows(
        outcomes,
        alpha0=args.alpha0,
        beta0=args.beta0,
        expected_n=args.expected_n,
    )
    output_path = Path(args.output_csv)
    write_full_posterior(output_path, rows)

    print(f"Wrote {len(rows)} full posterior distributions to {output_path}")
    for row in rows:
        print(
            f"  {row['model']}: "
            f"Beta({row['alpha_full']:g}, {row['beta_full']:g}), "
            f"n={row['n_full']}, accuracy={row['accuracy_full']:.4f}"
        )


if __name__ == "__main__":
    main()
