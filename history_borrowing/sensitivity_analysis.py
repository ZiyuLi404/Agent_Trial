"""
Alpha/lambda sensitivity analysis for history borrowing.

Evaluates every alpha/lambda grid point across all bucket-order permutations
for one or more named similarity sources.

Default inputs are the comparison outputs:
  history_borrowing/data/results/compare/all_orders/<source>/

Default outputs:
  history_borrowing/data/results/compare/alpha_lambda_sensitivity.csv
  history_borrowing/data/results/compare/alpha_lambda_sensitivity_summary.csv
  history_borrowing/data/results/compare/alpha_lambda_sensitivity.json
  history_borrowing/data/results/compare/alpha_lambda_sensitivity_<source>.png
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from train_borrow_params import compute_global_mae, load_model_distance, load_results, mean


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPARE_DIR = SCRIPT_DIR / "data" / "results" / "compare"

DEFAULT_SOURCES = {
    "fingerprint_conversation": DEFAULT_COMPARE_DIR / "all_orders" / "fingerprint_conversation",
    "embedding_diagnosis": DEFAULT_COMPARE_DIR / "all_orders" / "embedding_diagnosis",
    "hybrid_0.7_embedding_diagnosis_0.3_fingerprint_conversation": (
        DEFAULT_COMPARE_DIR / "all_orders" / "hybrid_0.7_embedding_diagnosis_0.3_fingerprint_conversation"
    ),
}

SOURCE_LABELS = {
    "fingerprint_conversation": "Fingerprint Conversation",
    "embedding_diagnosis": "Embedding Diagnosis",
    "hybrid_0.7_embedding_diagnosis_0.3_fingerprint_conversation": "Hybrid 0.7/0.3",
}


def parse_float_grid(raw: str) -> list[float]:
    try:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as e:
        sys.exit(f"ERROR parsing numeric grid '{raw}': {e}")


def parse_source_dirs(items: list[str] | None) -> dict[str, Path]:
    if not items:
        return dict(DEFAULT_SOURCES)

    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"ERROR: --source_dir must be NAME=DIR, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            sys.exit(f"ERROR: --source_dir has an empty source name: {item}")
        out[name] = Path(path)
    return out


def baseline_mae(perms: list[dict]) -> float:
    per_perm = [
        mean(abs(r["theta_25"] - r["theta_100"]) for r in perm["rows"])
        for perm in perms
    ]
    return float(mean(per_perm))


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8"):
            return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_filename(source: str) -> str:
    return source.replace("/", "_").replace(" ", "_")


def plot_heatmap(
    source: str,
    alpha_grid: list[float],
    lambda_grid: list[float],
    rows: list[dict],
    output_path: Path,
) -> None:
    mae_by_pair = {
        (float(row["alpha"]), float(row["lambda"])): float(row["borrowed_mae"])
        for row in rows
    }
    values = [
        [mae_by_pair[(alpha, lam)] for alpha in alpha_grid]
        for lam in lambda_grid
    ]

    best = min(rows, key=lambda r: float(r["borrowed_mae"]))
    best_alpha = float(best["alpha"])
    best_lambda = float(best["lambda"])
    best_alpha_i = alpha_grid.index(best_alpha)
    best_lambda_i = lambda_grid.index(best_lambda)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(values, cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(range(len(alpha_grid)))
    ax.set_xticklabels([f"{x:g}" for x in alpha_grid])
    ax.set_yticks(range(len(lambda_grid)))
    ax.set_yticklabels([f"{x:g}" for x in lambda_grid])
    ax.set_xlabel("alpha")
    ax.set_ylabel("lambda")

    label = SOURCE_LABELS.get(source, source)
    ax.set_title(f"Alpha/Lambda Sensitivity: {label}")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Borrowed MAE")

    for y, lam in enumerate(lambda_grid):
        for x, alpha in enumerate(alpha_grid):
            mae = mae_by_pair[(alpha, lam)]
            ax.text(x, y, f"{mae:.4f}", ha="center", va="center", fontsize=7)

    ax.add_patch(Rectangle(
        (best_alpha_i - 0.5, best_lambda_i - 0.5),
        1,
        1,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
    ))
    ax.text(
        best_alpha_i,
        best_lambda_i - 0.72,
        "best",
        ha="center",
        va="bottom",
        color="red",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate alpha/lambda sensitivity across history-borrowing sources."
    )
    parser.add_argument(
        "--source_dir",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help="Named all-orders source directory. May be passed multiple times.",
    )
    parser.add_argument("--alpha_grid", default="0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--lambda_grid", default="0,5,10,20,50,100,200")
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_COMPARE_DIR),
        help="Directory for sensitivity CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="alpha_lambda_sensitivity",
        help="Output filename prefix.",
    )
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    alpha_grid = parse_float_grid(args.alpha_grid)
    lambda_grid = parse_float_grid(args.lambda_grid)
    source_dirs = parse_source_dirs(args.source_dir)
    output_dir = Path(args.output_dir)

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    summary_json: dict = {
        "alpha_grid": alpha_grid,
        "lambda_grid": lambda_grid,
        "sources": {},
    }

    print(f"Grid: {len(alpha_grid)} alpha x {len(lambda_grid)} lambda = {len(alpha_grid) * len(lambda_grid)} points")

    for source, source_dir in source_dirs.items():
        if not source_dir.is_dir():
            sys.exit(f"ERROR: source directory not found for '{source}': {source_dir}")

        perms = load_results(source_dir)
        if not perms:
            sys.exit(f"ERROR: no valid results found for '{source}' in {source_dir}")

        model_dist = load_model_distance(source_dir)
        base_mae = baseline_mae(perms)
        source_rows: list[dict] = []

        for alpha in alpha_grid:
            for lam in lambda_grid:
                borrowed_mae = compute_global_mae(perms, model_dist, alpha, lam)
                improvement_abs = base_mae - borrowed_mae
                improvement_rel = improvement_abs / base_mae if base_mae > 0 else 0.0
                row = {
                    "similarity_source": source,
                    "alpha": alpha,
                    "lambda": lam,
                    "baseline_mae": round(base_mae, 6),
                    "borrowed_mae": round(borrowed_mae, 6),
                    "improvement_absolute": round(improvement_abs, 6),
                    "improvement_relative": round(improvement_rel, 6),
                }
                source_rows.append(row)

        best_row = min(source_rows, key=lambda r: float(r["borrowed_mae"]))
        for row in source_rows:
            row["is_best"] = (
                row["alpha"] == best_row["alpha"]
                and row["lambda"] == best_row["lambda"]
            )
        all_rows.extend(source_rows)

        summary = {
            "similarity_source": source,
            "best_alpha": best_row["alpha"],
            "best_lambda": best_row["lambda"],
            "baseline_mae": best_row["baseline_mae"],
            "best_borrowed_mae": best_row["borrowed_mae"],
            "improvement_absolute": best_row["improvement_absolute"],
            "improvement_relative": best_row["improvement_relative"],
            "n_permutations": len(perms),
            "n_grid_points": len(source_rows),
        }
        summary_rows.append(summary)
        summary_json["sources"][source] = summary

        if not args.no_plots:
            png_path = output_dir / f"{args.prefix}_{source_filename(source)}.png"
            plot_heatmap(source, alpha_grid, lambda_grid, source_rows, png_path)
            print(f"  {source}: best alpha={best_row['alpha']}, lambda={best_row['lambda']}, MAE={best_row['borrowed_mae']} -> {png_path}")
        else:
            print(f"  {source}: best alpha={best_row['alpha']}, lambda={best_row['lambda']}, MAE={best_row['borrowed_mae']}")

    full_csv = output_dir / f"{args.prefix}.csv"
    summary_csv = output_dir / f"{args.prefix}_summary.csv"
    json_path = output_dir / f"{args.prefix}.json"

    write_rows_csv(full_csv, all_rows)
    write_rows_csv(summary_csv, summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nFull sensitivity CSV : {full_csv}")
    print(f"Summary CSV          : {summary_csv}")
    print(f"Summary JSON         : {json_path}")


if __name__ == "__main__":
    main()
