#!/usr/bin/env python3
"""Render Algorithm 1 numeric and combined Algorithm 1/2 error tables."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-history-borrowing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_table(
    path: Path,
    title: str,
    headers: list[str],
    cells: list[list[str]],
    widths: list[float],
    highlight_last: bool = False,
) -> None:
    fig, axis = plt.subplots(figsize=(14.5, 1.05 + 0.82 * len(cells)))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=widths,
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
        elif highlight_last and row_index == len(cells):
            cell.set_facecolor("#EAF4EA")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#FAFBFD")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.04)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm1_results", required=True)
    parser.add_argument("--algorithm2_comparison", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--similarity_name", required=True)
    args = parser.parse_args()

    algorithm1 = read_csv(Path(args.algorithm1_results))
    algorithm2 = read_csv(Path(args.algorithm2_comparison))
    algorithm2_by_model = {row["model"]: row for row in algorithm2}
    if {row["model"] for row in algorithm1} != set(algorithm2_by_model):
        raise ValueError("Algorithm 1 and Algorithm 2 model sets do not match")

    numeric_rows = []
    error_rows = []
    for row in algorithm1:
        model = row["model"]
        distribution = algorithm2_by_model[model]
        bundle_value = float(row["theta_25"])
        borrowed_value = float(row["theta_borrowed"])
        full_value = float(row["theta_100"])
        numeric_rows.append(
            {
                "model": model,
                "bundle_n": int(distribution["fresh_n"]),
                "version_bundle_accuracy": bundle_value,
                "algorithm1_history_borrowed": borrowed_value,
                "full_data_accuracy": full_value,
            }
        )
        algorithm2_bundle_error = abs(
            float(distribution["fresh_mean"]) - float(distribution["full_mean"])
        )
        algorithm2_borrowed_error = abs(
            float(distribution["borrowed_mean"]) - float(distribution["full_mean"])
        )
        error_rows.append(
            {
                "model": model,
                "algorithm1_bundle_abs_error": abs(bundle_value - full_value),
                "algorithm1_borrowed_abs_error": abs(borrowed_value - full_value),
                "algorithm2_bundle_abs_error": algorithm2_bundle_error,
                "algorithm2_borrowed_abs_error": algorithm2_borrowed_error,
            }
        )

    mean_row = {
        "model": "Mean (MAE)",
        **{
            field: sum(float(row[field]) for row in error_rows) / len(error_rows)
            for field in list(error_rows[0])[1:]
        },
    }
    error_rows_with_mean = [*error_rows, mean_row]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "algorithm1_numeric_comparison.csv", numeric_rows)
    write_csv(output_dir / "combined_absolute_error.csv", error_rows_with_mean)

    numeric_lines = [
        "# Algorithm 1 version-bundle comparison",
        "",
        "| Model | Bundle n | Version-bundle accuracy | History borrowed | Full-data accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    numeric_cells = []
    for row in numeric_rows:
        values = [
            str(row["model"]),
            str(row["bundle_n"]),
            f"{row['version_bundle_accuracy']:.4f}",
            f"{row['algorithm1_history_borrowed']:.4f}",
            f"{row['full_data_accuracy']:.4f}",
        ]
        numeric_cells.append(values)
        numeric_lines.append(
            f"| {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} |"
        )
    (output_dir / "algorithm1_numeric_comparison.md").write_text(
        "\n".join(numeric_lines) + "\n", encoding="utf-8"
    )
    render_table(
        output_dir / "algorithm1_numeric_comparison.png",
        f"{args.dataset_name}: Algorithm 1 {args.similarity_name} comparison",
        [
            "Model",
            "Bundle\nn",
            "Version-bundle\naccuracy",
            "History-borrowed\nestimate",
            "Full-data\naccuracy",
        ],
        numeric_cells,
        [0.21, 0.09, 0.23, 0.25, 0.22],
    )

    error_lines = [
        "# Combined absolute-error comparison",
        "",
        "| Model | Algorithm 1 bundle | Algorithm 1 borrowed | Algorithm 2 bundle | Algorithm 2 borrowed |",
        "|---|---:|---:|---:|---:|",
    ]
    error_cells = []
    for row in error_rows_with_mean:
        values = [
            str(row["model"]),
            f"{row['algorithm1_bundle_abs_error']:.4f}",
            f"{row['algorithm1_borrowed_abs_error']:.4f}",
            f"{row['algorithm2_bundle_abs_error']:.4f}",
            f"{row['algorithm2_borrowed_abs_error']:.4f}",
        ]
        error_cells.append(values)
        error_lines.append(
            f"| {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} |"
        )
    error_lines.extend(
        [
            "",
            (
                "Algorithm 1 errors use full-data observed accuracy as the "
                "reference. Algorithm 2 errors use the full-data Beta posterior "
                "mean as the reference."
            ),
            "",
        ]
    )
    (output_dir / "combined_absolute_error.md").write_text(
        "\n".join(error_lines), encoding="utf-8"
    )
    render_table(
        output_dir / "combined_absolute_error.png",
        f"{args.dataset_name}: combined absolute error ({args.similarity_name})",
        [
            "Model",
            "Algorithm 1\nbundle",
            "Algorithm 1\nborrowed",
            "Algorithm 2\nbundle",
            "Algorithm 2\nborrowed",
        ],
        error_cells,
        [0.22, 0.195, 0.195, 0.195, 0.195],
        highlight_last=True,
    )
    print(f"Wrote numeric and combined error tables to {output_dir}")


if __name__ == "__main__":
    main()
