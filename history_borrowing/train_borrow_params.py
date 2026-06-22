"""
Global hyperparameter training for similarity-aware history borrowing.

Instead of tuning alpha/lambda per permutation, this script finds ONE shared
(alpha, lambda) pair per similarity source that minimises mean MAE across
ALL 24 bucket-order permutations and ALL 4 models simultaneously.

Inputs
------
  history_borrowing/history_borrowing_result/all_orders/diagnosis/
  history_borrowing/history_borrowing_result/all_orders/conversation/

Outputs
-------
  history_borrowing/borrow_params.json       -- best hyperparameters
  history_borrowing/borrow_params_summary.csv -- per-permutation summary
  history_borrowing/borrow_params_full.csv   -- long-form model-level detail

Usage
-----
  python history_borrowing/train_borrow_params.py

  python history_borrowing/train_borrow_params.py \
      --diagnosis_dir   history_borrowing/history_borrowing_result/all_orders/diagnosis \
      --conversation_dir history_borrowing/history_borrowing_result/all_orders/conversation \
      --alpha_grid "0.5,0.6,0.7,0.8,0.9,1.0" \
      --lambda_grid "0,5,10,20,50,100,200"
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> list[dict]:
    """
    Load all results_*.csv files from a directory.

    Returns a list of permutation dicts:
        [{
            "tag":        "b1_b2_b3_b4",
            "perm_order": ["bucket1","bucket2","bucket3","bucket4"],
            "rows": DataFrame with columns [model, assigned_bucket, theta_25, theta_100]
        }, ...]
    """
    csv_files = sorted(results_dir.glob("results_*.csv"))
    if not csv_files:
        print(f"  WARNING: no results_*.csv found in {results_dir}", file=sys.stderr)
        return []

    perms = []
    for f in csv_files:
        tag = f.stem.replace("results_", "")          # e.g. b1_b2_b3_b4
        parts = tag.split("_")                         # ['b1','b2','b3','b4']
        perm_order = [f"bucket{p[1:]}" for p in parts]  # ['bucket1','bucket2',...]

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}", file=sys.stderr)
            continue

        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"model", "assigned_bucket", "theta_25", "theta_100"}
        if not required.issubset(df.columns):
            print(
                f"  WARNING: {f.name} missing columns {required - set(df.columns)} — skipping.",
                file=sys.stderr,
            )
            continue

        perms.append({
            "tag": tag,
            "perm_order": perm_order,
            "rows": df[["model", "assigned_bucket", "theta_25", "theta_100"]].copy(),
        })

    return perms


def load_model_distance(summary_dir: Path) -> dict:
    """
    Load model_level_distance from any summary JSON in the directory.
    All permutations share the same distance matrix for a given similarity source.
    """
    summary_files = sorted(summary_dir.glob("summary_*.json"))
    if not summary_files:
        sys.exit(f"ERROR: no summary_*.json found in {summary_dir}")

    with open(summary_files[0]) as f:
        data = json.load(f)

    if "model_level_distance" not in data:
        sys.exit(f"ERROR: 'model_level_distance' key missing in {summary_files[0]}")

    return data["model_level_distance"]


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def compute_weights(target: str, models: list[str], model_dist: dict, lam: float) -> dict:
    """Softmax weights over peers using negative-distance kernel."""
    peers = [m for m in models if m != target]
    if not peers:
        return {}
    log_w = [-lam * model_dist[target][p] for p in peers]
    max_lw = max(log_w)
    exp_w = [math.exp(lw - max_lw) for lw in log_w]
    total = sum(exp_w)
    return {p: exp_w[i] / total for i, p in enumerate(peers)}


def compute_borrowed_estimate(
    theta_25: dict,
    model_dist: dict,
    alpha: float,
    lam: float,
) -> dict:
    """
    Return {model: theta_borrowed} for a single permutation.

    theta_25: {model: float}
    model_dist: {model_a: {model_b: float}}
    """
    models = list(theta_25.keys())
    result = {}
    for j in models:
        weights = compute_weights(j, models, model_dist, lam)
        peer = sum(weights[i] * theta_25[i] for i in weights)
        result[j] = alpha * theta_25[j] + (1.0 - alpha) * peer
    return result


# ---------------------------------------------------------------------------
# MAE computation
# ---------------------------------------------------------------------------

def compute_mae_single(perm: dict, model_dist: dict, alpha: float, lam: float) -> float:
    """MAE for one permutation at given (alpha, lambda)."""
    df = perm["rows"]
    theta_25 = dict(zip(df["model"], df["theta_25"]))
    theta_100 = dict(zip(df["model"], df["theta_100"]))
    borrowed = compute_borrowed_estimate(theta_25, model_dist, alpha, lam)
    errors = [abs(borrowed[m] - theta_100[m]) for m in theta_25]
    return float(np.mean(errors))


def compute_global_mae(
    perms: list[dict],
    model_dist: dict,
    alpha: float,
    lam: float,
) -> float:
    """Mean MAE across all permutations at given (alpha, lambda)."""
    maes = [compute_mae_single(p, model_dist, alpha, lam) for p in perms]
    return float(np.mean(maes))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search_params(
    perms: list[dict],
    model_dist: dict,
    alpha_grid: list[float],
    lambda_grid: list[float],
) -> tuple[float, float, float]:
    """
    Exhaustive grid search over (alpha, lambda).
    Returns (best_alpha, best_lambda, best_global_mae).
    """
    best_alpha, best_lam, best_mae = alpha_grid[0], lambda_grid[0], float("inf")
    for alpha in alpha_grid:
        for lam in lambda_grid:
            mae = compute_global_mae(perms, model_dist, alpha, lam)
            if mae < best_mae:
                best_mae, best_alpha, best_lam = mae, alpha, lam
    return best_alpha, best_lam, best_mae


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def build_perm_summary(
    source: str,
    perms: list[dict],
    model_dist: dict,
    alpha: float,
    lam: float,
) -> pd.DataFrame:
    """
    One row per permutation:
        source, permutation, mae_25, mae_borrowed, abs_improvement, rel_improvement
    """
    rows = []
    for perm in perms:
        df = perm["rows"]
        theta_25  = dict(zip(df["model"], df["theta_25"]))
        theta_100 = dict(zip(df["model"], df["theta_100"]))
        borrowed  = compute_borrowed_estimate(theta_25, model_dist, alpha, lam)

        mae_25  = float(np.mean([abs(theta_25[m]  - theta_100[m]) for m in theta_25]))
        mae_bor = float(np.mean([abs(borrowed[m]  - theta_100[m]) for m in theta_25]))
        imp_abs = mae_25 - mae_bor
        imp_rel = imp_abs / mae_25 if mae_25 > 0 else 0.0

        rows.append({
            "similarity_source":    source,
            "permutation":          perm["tag"],
            "alpha":                alpha,
            "lambda":               lam,
            "mae_25":               round(mae_25,  6),
            "mae_borrowed":         round(mae_bor, 6),
            "improvement_absolute": round(imp_abs, 6),
            "improvement_relative": round(imp_rel, 6),
        })
    return pd.DataFrame(rows)


def build_full_detail(
    source: str,
    perms: list[dict],
    model_dist: dict,
    alpha: float,
    lam: float,
) -> pd.DataFrame:
    """
    One row per (permutation × model):
        source, permutation, model, assigned_bucket,
        theta_25, theta_borrowed, theta_100,
        abs_error_25, abs_error_borrowed,
        alpha, lambda
    """
    rows = []
    for perm in perms:
        df = perm["rows"]
        theta_25  = dict(zip(df["model"], df["theta_25"]))
        theta_100 = dict(zip(df["model"], df["theta_100"]))
        bucket    = dict(zip(df["model"], df["assigned_bucket"]))
        borrowed  = compute_borrowed_estimate(theta_25, model_dist, alpha, lam)

        for m in theta_25:
            rows.append({
                "similarity_source":  source,
                "permutation":        perm["tag"],
                "model":              m,
                "assigned_bucket":    bucket[m],
                "theta_25":           round(theta_25[m], 4),
                "theta_borrowed":     round(borrowed[m], 4),
                "theta_100":          round(theta_100[m], 4),
                "abs_error_25":       round(abs(theta_25[m]  - theta_100[m]), 4),
                "abs_error_borrowed": round(abs(borrowed[m]  - theta_100[m]), 4),
                "alpha":              alpha,
                "lambda":             lam,
            })
    return pd.DataFrame(rows)


def per_model_mae_summary(full_df: pd.DataFrame) -> dict:
    """
    Average abs errors per model, per similarity source.
    Returns {source: {model: {mae_25, mae_borrowed}}}
    """
    out: dict = {}
    for source, sdf in full_df.groupby("similarity_source"):
        out[source] = {}
        for model, mdf in sdf.groupby("model"):
            out[source][model] = {
                "mae_25":       round(float(mdf["abs_error_25"].mean()), 6),
                "mae_borrowed": round(float(mdf["abs_error_borrowed"].mean()), 6),
            }
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_parameter_file(
    path: Path,
    best: dict,
    global_stats: dict,
    per_model: dict,
    perm_summary_df: pd.DataFrame,
) -> None:
    perm_records = perm_summary_df.to_dict(orient="records")
    payload = {**best, **global_stats, "per_model_mae": per_model, "per_permutation": perm_records}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def print_terminal_summary(
    best: dict,
    global_stats: dict,
    per_model: dict,
) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  GLOBAL HYPERPARAMETER TRAINING RESULTS")
    print(sep)
    print(f"  {'Source':<18} {'alpha':>6}  {'lambda':>7}  {'MAE_25':>8}  {'MAE_borrow':>11}  {'Improve%':>9}")
    print(f"  {'-'*64}")
    for src in ("diagnosis", "conversation"):
        a   = best.get(f"best_alpha_{src}", "—")
        l   = best.get(f"best_lambda_{src}", "—")
        m25 = global_stats.get(f"global_mae_25_{src}", float("nan"))
        mb  = global_stats.get(f"global_mae_borrowed_{src}", float("nan"))
        imp = (m25 - mb) / m25 * 100 if m25 > 0 else 0.0
        print(f"  {src:<18} {a:>6}  {l:>7}  {m25:>8.4f}  {mb:>11.4f}  {imp:>8.1f}%")
    print(sep)

    print("\n  Per-model MAE (averaged over all 24 permutations):\n")
    for src in ("diagnosis", "conversation"):
        print(f"  [{src}]")
        print(f"    {'Model':<25} {'MAE_25':>8}  {'MAE_borrow':>11}")
        print(f"    {'-'*48}")
        for model, vals in per_model.get(src, {}).items():
            m25 = vals["mae_25"]
            mb  = vals["mae_borrowed"]
            imp = (m25 - mb) / m25 * 100 if m25 > 0 else 0.0
            print(f"    {model:<25} {m25:>8.4f}  {mb:>11.4f}  ({imp:+.1f}%)")
        print()
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train global borrowing hyperparameters across all permutations."
    )
    parser.add_argument(
        "--diagnosis_dir",
        default="history_borrowing/history_borrowing_result/all_orders/diagnosis",
    )
    parser.add_argument(
        "--conversation_dir",
        default="history_borrowing/history_borrowing_result/all_orders/conversation",
    )
    parser.add_argument("--alpha_grid",  default="0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--lambda_grid", default="0,5,10,20,50,100,200")
    parser.add_argument(
        "--output_params",
        default="history_borrowing/borrow_params.json",
    )
    parser.add_argument(
        "--output_summary",
        default="history_borrowing/borrow_params_summary.csv",
    )
    parser.add_argument(
        "--output_full",
        default="history_borrowing/borrow_params_full.csv",
    )
    args = parser.parse_args()

    try:
        alpha_grid  = [float(x) for x in args.alpha_grid.split(",")]
        lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    except ValueError as e:
        sys.exit(f"ERROR parsing grids: {e}")

    # --- load ---
    sources = {
        "diagnosis":    Path(args.diagnosis_dir),
        "conversation": Path(args.conversation_dir),
    }

    all_perms: dict = {}
    all_dists: dict = {}
    missing_sources = []

    for src, d in sources.items():
        if not d.is_dir():
            print(f"  WARNING: directory not found — {d} (skipping '{src}')", file=sys.stderr)
            missing_sources.append(src)
            continue
        perms = load_results(d)
        dist  = load_model_distance(d)
        if not perms:
            print(f"  WARNING: no valid results in {d} (skipping '{src}')", file=sys.stderr)
            missing_sources.append(src)
            continue

        # Check for overlap if both sources present
        all_perms[src] = perms
        all_dists[src] = dist
        print(f"  Loaded {len(perms)} permutation(s) for '{src}'")

    if not all_perms:
        sys.exit("ERROR: no valid data found in either source directory.")

    if missing_sources:
        print(f"\n  NOTE: source(s) skipped due to missing data: {missing_sources}")

    # Check permutation coverage overlap
    tags_per_src = {src: {p["tag"] for p in perms} for src, perms in all_perms.items()}
    if len(tags_per_src) == 2:
        srcs = list(tags_per_src)
        common = tags_per_src[srcs[0]] & tags_per_src[srcs[1]]
        missing_in = {}
        for s in srcs:
            diff = tags_per_src[s] - common
            if diff:
                missing_in[s] = sorted(diff)
        if missing_in:
            print(f"\n  WARNING: permutation coverage mismatch — training on {len(common)} overlapping permutation(s).")
            for s, tags in missing_in.items():
                print(f"    Missing from '{s}': {tags}")
            # Restrict to common set
            for src in all_perms:
                all_perms[src] = [p for p in all_perms[src] if p["tag"] in common]

    # --- grid search ---
    n_combos = len(alpha_grid) * len(lambda_grid)
    print(f"\n  Grid: {len(alpha_grid)} alpha × {len(lambda_grid)} lambda = {n_combos} combinations")

    best: dict = {}
    global_stats: dict = {}
    perm_summary_dfs: list[pd.DataFrame] = []
    full_dfs: list[pd.DataFrame] = []

    for src, perms in all_perms.items():
        dist = all_dists[src]
        print(f"\n  [{src}] Grid-searching over {len(perms)} permutations ...")
        ba, bl, bm = grid_search_params(perms, dist, alpha_grid, lambda_grid)
        best[f"best_alpha_{src}"]  = ba
        best[f"best_lambda_{src}"] = bl

        # Baseline MAE (alpha=1 means no borrowing, but theta_25 is the baseline)
        mae_25_vals = [
            float(np.mean(np.abs(p["rows"]["theta_25"].values - p["rows"]["theta_100"].values)))
            for p in perms
        ]
        global_mae_25 = float(np.mean(mae_25_vals))
        global_stats[f"global_mae_25_{src}"]       = round(global_mae_25, 6)
        global_stats[f"global_mae_borrowed_{src}"]  = round(bm, 6)
        global_stats[f"global_improvement_abs_{src}"] = round(global_mae_25 - bm, 6)
        global_stats[f"global_improvement_rel_{src}"] = round(
            (global_mae_25 - bm) / global_mae_25 if global_mae_25 > 0 else 0.0, 6
        )

        perm_summary_dfs.append(build_perm_summary(src, perms, dist, ba, bl))
        full_dfs.append(build_full_detail(src, perms, dist, ba, bl))

    perm_summary_df = pd.concat(perm_summary_dfs, ignore_index=True)
    full_df         = pd.concat(full_dfs, ignore_index=True)

    # Per-model summary
    per_model = per_model_mae_summary(full_df)

    # --- save ---
    out_params  = Path(args.output_params)
    out_summary = Path(args.output_summary)
    out_full    = Path(args.output_full)

    save_parameter_file(out_params, best, global_stats, per_model, perm_summary_df)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    perm_summary_df.to_csv(out_summary, index=False)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_full, index=False)

    # --- terminal ---
    print_terminal_summary(best, global_stats, per_model)

    print(f"  Parameter file : {out_params}")
    print(f"  Summary CSV    : {out_summary}")
    print(f"  Full detail CSV: {out_full}\n")


if __name__ == "__main__":
    main()
