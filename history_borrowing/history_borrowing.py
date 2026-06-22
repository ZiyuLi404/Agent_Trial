"""
Similarity-aware history borrowing estimator for clinical LLM performance drift analysis.

Given:
  - accuracy_by_25_cases.csv  (per-bucket accuracies + total ground-truth)
  - similarity_matrix.csv     (pairwise cosine/embedding similarity between model replicates)

This script:
  1. Collapses replicate-level similarity into model-level similarity.
  2. Converts similarity to distance: d(A,B) = 1 - sim(A,B).
  3. Estimates each model's true accuracy by borrowing from its peers:

       theta_borrowed_j = alpha * theta_25_j
                        + (1 - alpha) * sum_{i != j} w_ij * theta_25_i

     where w_ij = exp(-lambda * d(i,j)) / sum_{i'!=j} exp(-lambda * d(i',j))

  4. Grid-searches (alpha, lambda) to minimise MAE vs. ground-truth total accuracy.

Usage:
    python history_borrowing/history_borrowing.py \
        --accuracy_csv history_borrowing/accuracy_by_25_cases.csv \
        --similarity_csv history_borrowing/similarity_matrix.csv \
        --replicate_map_json history_borrowing/replicate_map.json \
        --output_csv history_borrowing/history_borrowing_results.csv \
        --output_summary_json history_borrowing/history_borrowing_summary.json
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_accuracy_csv(path: str) -> tuple[list[str], dict, dict, list[str]]:
    """
    Returns:
        models       -- ordered list of model names
        theta_25     -- {model: float}  (per-bucket accuracy for assigned bucket)
        theta_100    -- {model: float}  (total ground-truth accuracy)
        bucket_cols  -- list of bucket column names found in the CSV
    """
    models = []
    theta_25_raw: dict = {}   # model -> {col: val}
    theta_100: dict = {}
    bucket_cols: list[str] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"ERROR: empty CSV at {path}")
        bucket_cols = [c for c in reader.fieldnames if c.startswith("bucket")]
        for row in reader:
            name = row["model"].strip()
            models.append(name)
            try:
                theta_100[name] = float(row["total"])
            except (ValueError, KeyError):
                sys.exit(f"ERROR: missing or non-numeric 'total' for model '{name}' in {path}")
            theta_25_raw[name] = {}
            for bc in bucket_cols:
                try:
                    theta_25_raw[name][bc] = float(row[bc])
                except (ValueError, KeyError):
                    theta_25_raw[name][bc] = None

    return models, theta_25_raw, theta_100, bucket_cols


def load_similarity_csv(path: str) -> tuple[list[str], dict]:
    """
    Returns:
        labels      -- ordered list of label names (row/column headers)
        sim_matrix  -- {label_i: {label_j: float}}
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"ERROR: empty CSV at {path}")
        # First column is the row-label column (named 'model' or similar).
        row_label_col = reader.fieldnames[0]
        labels = [c for c in reader.fieldnames if c != row_label_col]
        sim_matrix: dict = {}
        for row in reader:
            src = row[row_label_col].strip()
            sim_matrix[src] = {}
            for lbl in labels:
                try:
                    sim_matrix[src][lbl] = float(row[lbl])
                except (ValueError, KeyError):
                    sim_matrix[src][lbl] = float("nan")

    return labels, sim_matrix


# ---------------------------------------------------------------------------
# Similarity collapsing
# ---------------------------------------------------------------------------

def collapse_similarity_matrix(
    labels: list[str],
    sim_matrix: dict,
    replicate_map: dict,
) -> tuple[dict, dict]:
    """
    Collapse replicate-level similarity into model-level similarity and distance.

    replicate_map: {model_name: [label1, label2, ...]}

    Returns:
        model_sim  -- {model_A: {model_B: float}}
        model_dist -- {model_A: {model_B: float}}  (1 - sim)
    """
    models_m = list(replicate_map.keys())
    model_sim: dict = {m: {} for m in models_m}
    model_dist: dict = {m: {} for m in models_m}

    for i, m_a in enumerate(models_m):
        for j, m_b in enumerate(models_m):
            labels_a = replicate_map[m_a]
            labels_b = replicate_map[m_b]
            pairs = []
            for la in labels_a:
                for lb in labels_b:
                    if la in sim_matrix and lb in sim_matrix[la]:
                        v = sim_matrix[la][lb]
                        if not math.isnan(v):
                            pairs.append(v)
            if not pairs:
                sys.exit(
                    f"ERROR: no valid similarity values found between "
                    f"'{m_a}' labels {labels_a} and '{m_b}' labels {labels_b}."
                )
            avg_sim = sum(pairs) / len(pairs)
            model_sim[m_a][m_b] = avg_sim
            model_dist[m_a][m_b] = 1.0 - avg_sim

    return model_sim, model_dist


def build_identity_replicate_map(models: list[str], labels: list[str]) -> dict:
    """
    If no replicate_map is provided, try to match model names directly to similarity labels.
    Raises an informative error if a model name is missing from the similarity matrix.
    """
    missing = [m for m in models if m not in labels]
    if missing:
        sys.exit(
            f"ERROR: the following model name(s) are not present in the similarity matrix "
            f"column headers: {missing}\n"
            "Provide --replicate_map_json to map model names to their similarity-matrix labels."
        )
    return {m: [m] for m in models}


# ---------------------------------------------------------------------------
# Bucket assignment
# ---------------------------------------------------------------------------

def select_theta_25(
    models: list[str],
    theta_25_raw: dict,
    bucket_cols: list[str],
    bucket_assignment: dict,
) -> dict:
    """
    Return {model: float} by picking each model's assigned bucket value.

    bucket_assignment: {model: "bucket1" | "bucket2" | ...}
    """
    theta_25: dict = {}
    for model in models:
        bc = bucket_assignment.get(model)
        if bc is None:
            sys.exit(f"ERROR: no bucket assignment found for model '{model}'.")
        val = theta_25_raw[model].get(bc)
        if val is None:
            sys.exit(
                f"ERROR: model '{model}' has no value for assigned bucket '{bc}'. "
                "Check --accuracy_csv and --bucket_assignment."
            )
        theta_25[model] = val
    return theta_25


def default_bucket_assignment(models: list[str], bucket_cols: list[str]) -> dict:
    """Assign models to buckets by row order (model[0]->bucket1, etc.)."""
    if len(models) > len(bucket_cols):
        sys.exit(
            f"ERROR: {len(models)} models but only {len(bucket_cols)} bucket column(s). "
            "Cannot assign one bucket per model by row order."
        )
    return {m: bucket_cols[i] for i, m in enumerate(models)}


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def compute_weights(model: str, models: list[str], model_dist: dict, lam: float) -> dict:
    """
    Softmax weights over peers (all models except `model`) using negative-distance kernel.
    Returns {peer: weight}.
    """
    peers = [m for m in models if m != model]
    if not peers:
        return {}

    log_w = [-lam * model_dist[model][p] for p in peers]
    # Numerically stable softmax
    max_lw = max(log_w)
    exp_w = [math.exp(lw - max_lw) for lw in log_w]
    total = sum(exp_w)
    return {p: exp_w[i] / total for i, p in enumerate(peers)}


def compute_borrowed_estimates(
    models: list[str],
    theta_25: dict,
    model_dist: dict,
    alpha: float,
    lam: float,
) -> dict:
    """Return {model: theta_borrowed} for given alpha and lambda."""
    estimates: dict = {}
    for j in models:
        weights = compute_weights(j, models, model_dist, lam)
        peer_contribution = sum(weights[i] * theta_25[i] for i in weights)
        estimates[j] = alpha * theta_25[j] + (1.0 - alpha) * peer_contribution
    return estimates


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search_alpha_lambda(
    models: list[str],
    theta_25: dict,
    theta_100: dict,
    model_dist: dict,
    alpha_grid: list[float],
    lambda_grid: list[float],
) -> tuple[float, float, float]:
    """
    Minimise MAE(alpha, lambda) = mean_j |theta_borrowed_j - theta_100_j|.
    Returns (best_alpha, best_lambda, best_mae).
    """
    best_alpha = alpha_grid[0]
    best_lam = lambda_grid[0]
    best_mae = float("inf")

    for alpha in alpha_grid:
        for lam in lambda_grid:
            estimates = compute_borrowed_estimates(models, theta_25, model_dist, alpha, lam)
            mae = sum(abs(estimates[j] - theta_100[j]) for j in models) / len(models)
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
                best_lam = lam

    return best_alpha, best_lam, best_mae


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(
    models: list[str],
    bucket_assignment: dict,
    theta_25: dict,
    theta_borrowed: dict,
    theta_100: dict,
    best_alpha: float,
    best_lam: float,
    model_sim: dict,
    model_dist: dict,
    output_csv: str,
    output_summary_json: str,
) -> None:
    mae_25 = sum(abs(theta_25[j] - theta_100[j]) for j in models) / len(models)
    mae_borrowed = sum(abs(theta_borrowed[j] - theta_100[j]) for j in models) / len(models)
    improvement_absolute = mae_25 - mae_borrowed
    improvement_relative = improvement_absolute / mae_25 if mae_25 > 0 else 0.0

    # CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "assigned_bucket",
            "theta_25", "theta_borrowed", "theta_100",
            "error_25", "error_borrowed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for j in models:
            writer.writerow({
                "model": j,
                "assigned_bucket": bucket_assignment[j],
                "theta_25": round(theta_25[j], 4),
                "theta_borrowed": round(theta_borrowed[j], 4),
                "theta_100": round(theta_100[j], 4),
                "error_25": round(theta_25[j] - theta_100[j], 4),
                "error_borrowed": round(theta_borrowed[j] - theta_100[j], 4),
            })

    # JSON summary
    summary = {
        "best_alpha": best_alpha,
        "best_lambda": best_lam,
        "mae_25": round(mae_25, 6),
        "mae_borrowed": round(mae_borrowed, 6),
        "improvement_absolute": round(improvement_absolute, 6),
        "improvement_relative": round(improvement_relative, 6),
        "model_level_similarity": {
            a: {b: round(model_sim[a][b], 6) for b in model_sim[a]}
            for a in model_sim
        },
        "model_level_distance": {
            a: {b: round(model_dist[a][b], 6) for b in model_dist[a]}
            for a in model_dist
        },
    }
    Path(output_summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Terminal summary
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Best alpha  : {best_alpha}")
    print(f"  Best lambda : {best_lam}")
    print(f"  MAE (theta_25 baseline) : {mae_25:.4f}")
    print(f"  MAE (history borrowing) : {mae_borrowed:.4f}")
    improved = improvement_absolute > 1e-9
    print(f"  Borrowing {'IMPROVED' if improved else 'DID NOT IMPROVE'} accuracy estimation")
    if improved:
        print(f"  Absolute improvement    : {improvement_absolute:.4f}")
        print(f"  Relative improvement    : {improvement_relative*100:.2f}%")
    print(f"{sep}")

    print(f"\n{'Model':<25} {'Bucket':>8} {'θ_25':>7} {'θ_borrow':>9} {'θ_100':>7} "
          f"{'err_25':>8} {'err_borrow':>11}")
    print("-" * 80)
    for j in models:
        print(
            f"{j:<25} {bucket_assignment[j]:>8} "
            f"{theta_25[j]:>7.4f} {theta_borrowed[j]:>9.4f} {theta_100[j]:>7.4f} "
            f"{theta_25[j]-theta_100[j]:>+8.4f} {theta_borrowed[j]-theta_100[j]:>+11.4f}"
        )
    print(f"\nResults CSV   : {output_csv}")
    print(f"Summary JSON  : {output_summary_json}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Similarity-aware history borrowing for LLM performance estimation."
    )
    parser.add_argument(
        "--accuracy_csv",
        default="history_borrowing/accuracy_by_25_cases.csv",
        help="Accuracy CSV produced by accuracy_summary.py.",
    )
    parser.add_argument(
        "--similarity_csv",
        default="history_borrowing/similarity_matrix/diagnosis_similarity_matrix.csv",
        help=(
            "Pairwise similarity matrix CSV "
            "(default: history_borrowing/similarity_matrix/diagnosis_similarity_matrix.csv). "
            "Use e.g. --similarity_csv history_borrowing/similarity_matrix/conversation_similarity_matrix.csv "
            "to switch matrices."
        ),
    )
    parser.add_argument(
        "--replicate_map_json",
        default=None,
        help=(
            "Optional JSON file mapping model names to their similarity-matrix labels. "
            'E.g. {"deepseek-v4-flash": ["flash_1", "flash_2"], ...}'
        ),
    )
    parser.add_argument(
        "--output_csv",
        default="history_borrowing/history_borrowing_result/history_borrowing_results.csv",
        help="Output CSV with per-model results (default: history_borrowing/history_borrowing_result/).",
    )
    parser.add_argument(
        "--output_summary_json",
        default="history_borrowing/history_borrowing_result/history_borrowing_summary.json",
        help="Output JSON with best hyperparameters and MAE summary (default: history_borrowing/history_borrowing_result/).",
    )
    parser.add_argument(
        "--alpha_grid",
        default="0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated alpha values to search (default: 0.5,0.6,0.7,0.8,0.9,1.0).",
    )
    parser.add_argument(
        "--lambda_grid",
        default="0,5,10,20,50,100,200",
        help="Comma-separated lambda values to search (default: 0,5,10,20,50,100,200).",
    )
    parser.add_argument(
        "--bucket_order",
        default=None,
        help=(
            "Comma-separated bucket names in model-row order. "
            "Simpler alternative to --bucket_assignment. "
            "E.g. --bucket_order bucket2,bucket1,bucket3,bucket4 assigns "
            "the first model to bucket2, the second to bucket1, etc."
        ),
    )
    parser.add_argument(
        "--bucket_assignment",
        default=None,
        help=(
            "Optional JSON string or file path assigning each model to a bucket by name. "
            'E.g. \'{"deepseek-v4-flash":"bucket2","deepseek-v4-pro":"bucket1",...}\'. '
            "Use --bucket_order for a simpler positional syntax."
        ),
    )
    args = parser.parse_args()

    if args.bucket_order and args.bucket_assignment:
        sys.exit("ERROR: supply either --bucket_order or --bucket_assignment, not both.")

    # --- parse grids ---
    try:
        alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
        lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    except ValueError as e:
        sys.exit(f"ERROR parsing alpha_grid or lambda_grid: {e}")

    # --- load accuracy CSV ---
    models, theta_25_raw, theta_100, bucket_cols = load_accuracy_csv(args.accuracy_csv)
    if not models:
        sys.exit(f"ERROR: no model rows found in {args.accuracy_csv}")

    # --- load similarity CSV ---
    sim_labels, sim_matrix = load_similarity_csv(args.similarity_csv)

    # --- replicate map ---
    if args.replicate_map_json:
        rmap_path = Path(args.replicate_map_json)
        if rmap_path.exists():
            with open(rmap_path, encoding="utf-8") as f:
                replicate_map = json.load(f)
        else:
            # Maybe it was passed as an inline JSON string
            try:
                replicate_map = json.loads(args.replicate_map_json)
            except json.JSONDecodeError:
                sys.exit(
                    f"ERROR: --replicate_map_json is neither a valid file path "
                    f"nor valid JSON: {args.replicate_map_json}"
                )
        # Validate: all models must be present in replicate_map
        missing = [m for m in models if m not in replicate_map]
        if missing:
            sys.exit(
                f"ERROR: the following model(s) from accuracy CSV are not in replicate_map: "
                f"{missing}"
            )
        # Validate: all labels must be present in similarity matrix
        for model_name, lbls in replicate_map.items():
            bad = [l for l in lbls if l not in sim_matrix]
            if bad:
                sys.exit(
                    f"ERROR: replicate labels {bad} for model '{model_name}' "
                    f"are not found in similarity matrix rows."
                )
    else:
        replicate_map = build_identity_replicate_map(models, sim_labels)

    # Restrict replicate_map to models present in accuracy CSV (preserve order)
    replicate_map = {m: replicate_map[m] for m in models if m in replicate_map}

    # --- collapse similarity ---
    model_sim, model_dist = collapse_similarity_matrix(sim_labels, sim_matrix, replicate_map)

    # --- bucket assignment ---
    if args.bucket_order:
        buckets = [b.strip() for b in args.bucket_order.split(",")]
        if len(buckets) != len(models):
            sys.exit(
                f"ERROR: --bucket_order has {len(buckets)} value(s) but there are "
                f"{len(models)} model(s) in the accuracy CSV. "
                f"Models (in order): {models}"
            )
        bad = [b for b in buckets if b not in bucket_cols]
        if bad:
            sys.exit(
                f"ERROR: unknown bucket name(s) in --bucket_order: {bad}. "
                f"Available: {bucket_cols}"
            )
        bucket_assignment = dict(zip(models, buckets))
    elif args.bucket_assignment:
        ba_arg = args.bucket_assignment.strip()
        ba_path = Path(ba_arg)
        if ba_path.exists():
            with open(ba_path, encoding="utf-8") as f:
                bucket_assignment = json.load(f)
        else:
            try:
                bucket_assignment = json.loads(ba_arg)
            except json.JSONDecodeError:
                sys.exit(
                    f"ERROR: --bucket_assignment is neither a valid file path "
                    f"nor valid JSON: {ba_arg}"
                )
        missing = [m for m in models if m not in bucket_assignment]
        if missing:
            sys.exit(
                f"ERROR: the following model(s) have no bucket assignment: {missing}"
            )
    else:
        bucket_assignment = default_bucket_assignment(models, bucket_cols)

    print(f"\nBucket assignment:")
    for m, bc in bucket_assignment.items():
        print(f"  {m} -> {bc}")

    # --- select theta_25 ---
    theta_25 = select_theta_25(models, theta_25_raw, bucket_cols, bucket_assignment)

    # --- grid search ---
    print(
        f"\nGrid search: {len(alpha_grid)} alpha × {len(lambda_grid)} lambda "
        f"= {len(alpha_grid)*len(lambda_grid)} combinations ..."
    )
    best_alpha, best_lam, best_mae = grid_search_alpha_lambda(
        models, theta_25, theta_100, model_dist, alpha_grid, lambda_grid
    )

    # --- final estimates with best hyperparameters ---
    theta_borrowed = compute_borrowed_estimates(
        models, theta_25, model_dist, best_alpha, best_lam
    )

    # --- save ---
    save_outputs(
        models=models,
        bucket_assignment=bucket_assignment,
        theta_25=theta_25,
        theta_borrowed=theta_borrowed,
        theta_100=theta_100,
        best_alpha=best_alpha,
        best_lam=best_lam,
        model_sim=model_sim,
        model_dist=model_dist,
        output_csv=args.output_csv,
        output_summary_json=args.output_summary_json,
    )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Example commands
# ---------------------------------------------------------------------------
#
# python history_borrowing/accuracy_summary.py \
#   --groundtruth_dir history_borrowing/groundtruth \
#   --bucket_size 25 \
#   --output_csv history_borrowing/accuracy_by_25_cases.csv
#
# Default (diagnosis matrix, row-order bucket assignment):
# python history_borrowing/history_borrowing.py \
#   --accuracy_csv history_borrowing/accuracy_by_25_cases.csv \
#   --replicate_map_json history_borrowing/replicate_map.json
#
# Reverse bucket order (model1->bucket2, model2->bucket1, model3->bucket4, model4->bucket3):
# python history_borrowing/history_borrowing.py \
#   --accuracy_csv history_borrowing/accuracy_by_25_cases.csv \
#   --replicate_map_json history_borrowing/replicate_map.json \
#   --bucket_order bucket2,bucket1,bucket4,bucket3
#
# Explicit per-model bucket assignment:
# python history_borrowing/history_borrowing.py \
#   --accuracy_csv history_borrowing/accuracy_by_25_cases.csv \
#   --replicate_map_json history_borrowing/replicate_map.json \
#   --bucket_assignment '{"deepseek-v4-flash":"bucket2","deepseek-v4-pro":"bucket1","gpt-5_4-mini":"bucket4","gpt-5_5":"bucket3"}'
#
# With conversation similarity matrix:
# python history_borrowing/history_borrowing.py \
#   --accuracy_csv history_borrowing/accuracy_by_25_cases.csv \
#   --similarity_csv history_borrowing/similarity_matrix/conversation_similarity_matrix.csv \
#   --replicate_map_json history_borrowing/replicate_map.json \
#   --output_csv history_borrowing/history_borrowing_result/conversation_results.csv \
#   --output_summary_json history_borrowing/history_borrowing_result/conversation_summary.json
