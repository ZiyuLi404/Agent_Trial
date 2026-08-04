#!/usr/bin/env python3
"""Create Algorithm 1 accuracy inputs from balanced version bundles."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from history_borrowing.algorithm_2.bayesian_pseudo_posterior import (
    load_full_model_outcomes,
    load_model_outcome_buckets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundtruth_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    groundtruth_dir = Path(args.groundtruth_dir)
    buckets_by_model = load_model_outcome_buckets(groundtruth_dir)
    full_by_model = {
        outcome.model: outcome for outcome in load_full_model_outcomes(groundtruth_dir)
    }
    bucket_count = len(buckets_by_model)
    if any(len(buckets) != bucket_count for buckets in buckets_by_model.values()):
        raise ValueError("Expected one version bundle per model version")

    rows = []
    for model, buckets in buckets_by_model.items():
        full = full_by_model[model]
        row = {
            "model": model,
            "total": full.mean_reward,
        }
        row.update(
            {
                f"bucket{index + 1}": bundle.mean_reward
                for index, bundle in enumerate(buckets)
            }
        )
        rows.append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "total",
                *[f"bucket{index + 1}" for index in range(bucket_count)],
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    sizes = [bundle.n for bundle in next(iter(buckets_by_model.values()))]
    print(
        f"Wrote {len(rows)} models with version bundle sizes {sizes} to {output_path}"
    )


if __name__ == "__main__":
    main()
