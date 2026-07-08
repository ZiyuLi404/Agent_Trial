"""
Global hyperparameter training for similarity-aware history borrowing.

Instead of tuning alpha/lambda per permutation, this script finds ONE shared
(alpha, lambda) pair per similarity source that minimises mean MAE across
ALL 24 bucket-order permutations and ALL 4 models simultaneously.

Inputs
------
  history_borrowing/data/results/all_orders/diagnosis/
  history_borrowing/data/results/all_orders/conversation/
  history_borrowing/data/results/all_orders/fingerprint_conversation/

Outputs
-------
  history_borrowing/data/results/borrow_params.json        -- best hyperparameters
  history_borrowing/data/results/borrow_params_summary.csv -- per-permutation summary
  history_borrowing/data/results/borrow_params_full.csv    -- long-form model-level detail

Usage
-----
  python history_borrowing/train_borrow_params.py

  python history_borrowing/train_borrow_params.py \
      --diagnosis_dir   history_borrowing/data/results/all_orders/diagnosis \
      --conversation_dir history_borrowing/data/results/all_orders/conversation \
      --fingerprint_conversation_dir history_borrowing/data/results/all_orders/fingerprint_conversation \
      --alpha_grid "0.5,0.6,0.7,0.8,0.9,1.0" \
      --lambda_grid "0,5,10,20,50,100,200"

  python history_borrowing/train_borrow_params.py \
      --source_dir fingerprint_conversation=history_borrowing/data/results/compare/all_orders/fingerprint_conversation \
      --source_dir embedding_diagnosis=history_borrowing/data/results/compare/all_orders/embedding_diagnosis \
      --source_dir hybrid_0.7_embedding_diagnosis_0.3_fingerprint_conversation=history_borrowing/data/results/compare/all_orders/hybrid_0.7_embedding_diagnosis_0.3_fingerprint_conversation
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

def load_results(results_dir: Path) -> list[dict]:
    """
    Load all results_*.csv files from a directory.

    Returns a list of permutation dicts:
        [{
            "tag":        "b1_b2_b3_b4",
            "perm_order": ["bucket1","bucket2","bucket3","bucket4"],
            "rows": list of rows with model, assigned_bucket, theta_25, theta_100
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

        required = {"model", "assigned_bucket", "theta_25", "theta_100"}
        rows = []
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    print(f"  WARNING: {f.name} is empty — skipping.", file=sys.stderr)
                    continue

                fieldnames = {c.strip().lower() for c in reader.fieldnames}
                if not required.issubset(fieldnames):
                    print(
                        f"  WARNING: {f.name} missing columns {required - fieldnames} — skipping.",
                        file=sys.stderr,
                    )
                    continue

                for raw in reader:
                    row = {k.strip().lower(): v for k, v in raw.items()}
                    rows.append({
                        "model": row["model"],
                        "assigned_bucket": row["assigned_bucket"],
                        "theta_25": float(row["theta_25"]),
                        "theta_100": float(row["theta_100"]),
                    })
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}", file=sys.stderr)
            continue

        perms.append({
            "tag": tag,
            "perm_order": perm_order,
            "rows": rows,
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

def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def compute_mae_single(perm: dict, model_dist: dict, alpha: float, lam: float) -> float:
    """MAE for one permutation at given (alpha, lambda)."""
    rows = perm["rows"]
    theta_25 = {r["model"]: r["theta_25"] for r in rows}
    theta_100 = {r["model"]: r["theta_100"] for r in rows}
    borrowed = compute_borrowed_estimate(theta_25, model_dist, alpha, lam)
    errors = [abs(borrowed[m] - theta_100[m]) for m in theta_25]
    return float(mean(errors))


def compute_global_mae(
    perms: list[dict],
    model_dist: dict,
    alpha: float,
    lam: float,
) -> float:
    """Mean MAE across all permutations at given (alpha, lambda)."""
    maes = [compute_mae_single(p, model_dist, alpha, lam) for p in perms]
    return float(mean(maes))


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
) -> list[dict]:
    """
    One row per permutation:
        source, permutation, mae_25, mae_borrowed, abs_improvement, rel_improvement
    """
    rows = []
    for perm in perms:
        perm_rows = perm["rows"]
        theta_25  = {r["model"]: r["theta_25"] for r in perm_rows}
        theta_100 = {r["model"]: r["theta_100"] for r in perm_rows}
        borrowed  = compute_borrowed_estimate(theta_25, model_dist, alpha, lam)

        mae_25  = float(mean(abs(theta_25[m]  - theta_100[m]) for m in theta_25))
        mae_bor = float(mean(abs(borrowed[m]  - theta_100[m]) for m in theta_25))
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
    return rows


def build_full_detail(
    source: str,
    perms: list[dict],
    model_dist: dict,
    alpha: float,
    lam: float,
) -> list[dict]:
    """
    One row per (permutation × model):
        source, permutation, model, assigned_bucket,
        theta_25, theta_borrowed, theta_100,
        abs_error_25, abs_error_borrowed,
        alpha, lambda
    """
    rows = []
    for perm in perms:
        perm_rows = perm["rows"]
        theta_25  = {r["model"]: r["theta_25"] for r in perm_rows}
        theta_100 = {r["model"]: r["theta_100"] for r in perm_rows}
        bucket    = {r["model"]: r["assigned_bucket"] for r in perm_rows}
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
    return rows


def per_model_mae_summary(full_rows: list[dict]) -> dict:
    """
    Average abs errors per model, per similarity source.
    Returns {source: {model: {mae_25, mae_borrowed}}}
    """
    out: dict = {}
    grouped: dict = {}
    for row in full_rows:
        grouped.setdefault(row["similarity_source"], {}).setdefault(row["model"], []).append(row)

    for source, models in grouped.items():
        out[source] = {}
        for model, rows in models.items():
            out[source][model] = {
                "mae_25": round(float(mean(r["abs_error_25"] for r in rows)), 6),
                "mae_borrowed": round(float(mean(r["abs_error_borrowed"] for r in rows)), 6),
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
    perm_summary_rows: list[dict],
) -> None:
    payload = {**best, **global_stats, "per_model_mae": per_model, "per_permutation": perm_summary_rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8"):
            return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_terminal_summary(
    best: dict,
    global_stats: dict,
    per_model: dict,
    sources: list[str],
) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  GLOBAL HYPERPARAMETER TRAINING RESULTS")
    print(sep)
    print(f"  {'Source':<18} {'alpha':>6}  {'lambda':>7}  {'MAE_25':>8}  {'MAE_borrow':>11}  {'Improve%':>9}")
    print(f"  {'-'*64}")
    for src in sources:
        a   = best.get(f"best_alpha_{src}", "—")
        l   = best.get(f"best_lambda_{src}", "—")
        m25 = global_stats.get(f"global_mae_25_{src}", float("nan"))
        mb  = global_stats.get(f"global_mae_borrowed_{src}", float("nan"))
        imp = (m25 - mb) / m25 * 100 if m25 > 0 else 0.0
        print(f"  {src:<18} {a:>6}  {l:>7}  {m25:>8.4f}  {mb:>11.4f}  {imp:>8.1f}%")
    print(sep)

    print("\n  Per-model MAE (averaged over all 24 permutations):\n")
    for src in sources:
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
        default="history_borrowing/data/results/all_orders/diagnosis",
    )
    parser.add_argument(
        "--conversation_dir",
        default="history_borrowing/data/results/all_orders/conversation",
    )
    parser.add_argument(
        "--fingerprint_conversation_dir",
        default="history_borrowing/data/results/all_orders/fingerprint_conversation",
    )
    parser.add_argument(
        "--source_dir",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help=(
            "Optional named source directory. May be passed multiple times. "
            "When supplied, these sources replace the built-in diagnosis/conversation/"
            "fingerprint_conversation defaults."
        ),
    )
    parser.add_argument("--alpha_grid",  default="0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--lambda_grid", default="0,5,10,20,50,100,200")
    parser.add_argument(
        "--output_params",
        default="history_borrowing/data/results/borrow_params.json",
    )
    parser.add_argument(
        "--output_summary",
        default="history_borrowing/data/results/borrow_params_summary.csv",
    )
    parser.add_argument(
        "--output_full",
        default="history_borrowing/data/results/borrow_params_full.csv",
    )
    args = parser.parse_args()

    try:
        alpha_grid  = [float(x) for x in args.alpha_grid.split(",")]
        lambda_grid = [float(x) for x in args.lambda_grid.split(",")]
    except ValueError as e:
        sys.exit(f"ERROR parsing grids: {e}")

    # --- load ---
    if args.source_dir:
        sources = {}
        for item in args.source_dir:
            if "=" not in item:
                sys.exit(f"ERROR: --source_dir must be NAME=DIR, got: {item}")
            name, path = item.split("=", 1)
            name = name.strip()
            if not name:
                sys.exit(f"ERROR: --source_dir has an empty source name: {item}")
            sources[name] = Path(path)
    else:
        sources = {
            "diagnosis":    Path(args.diagnosis_dir),
            "conversation": Path(args.conversation_dir),
            "fingerprint_conversation": Path(args.fingerprint_conversation_dir),
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

        all_perms[src] = perms
        all_dists[src] = dist
        print(f"  Loaded {len(perms)} permutation(s) for '{src}'")

    if not all_perms:
        sys.exit("ERROR: no valid data found in either source directory.")

    if missing_sources:
        print(f"\n  NOTE: source(s) skipped due to missing data: {missing_sources}")

    # Check permutation coverage overlap across all loaded sources.
    tags_per_src = {src: {p["tag"] for p in perms} for src, perms in all_perms.items()}
    if len(tags_per_src) > 1:
        common = set.intersection(*tags_per_src.values())
        union = set.union(*tags_per_src.values())
        if not common:
            sys.exit("ERROR: no overlapping bucket-order permutations across loaded sources.")
        missing_in = {}
        for s, tags in tags_per_src.items():
            missing = union - tags
            if missing:
                missing_in[s] = sorted(missing)
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
    perm_summary_rows: list[dict] = []
    full_rows: list[dict] = []

    for src, perms in all_perms.items():
        dist = all_dists[src]
        print(f"\n  [{src}] Grid-searching over {len(perms)} permutations ...")
        ba, bl, bm = grid_search_params(perms, dist, alpha_grid, lambda_grid)
        best[f"best_alpha_{src}"]  = ba
        best[f"best_lambda_{src}"] = bl

        # Baseline MAE (alpha=1 means no borrowing, but theta_25 is the baseline)
        mae_25_vals = [
            float(mean(abs(r["theta_25"] - r["theta_100"]) for r in p["rows"]))
            for p in perms
        ]
        global_mae_25 = float(mean(mae_25_vals))
        global_stats[f"global_mae_25_{src}"]       = round(global_mae_25, 6)
        global_stats[f"global_mae_borrowed_{src}"]  = round(bm, 6)
        global_stats[f"global_improvement_abs_{src}"] = round(global_mae_25 - bm, 6)
        global_stats[f"global_improvement_rel_{src}"] = round(
            (global_mae_25 - bm) / global_mae_25 if global_mae_25 > 0 else 0.0, 6
        )

        perm_summary_rows.extend(build_perm_summary(src, perms, dist, ba, bl))
        full_rows.extend(build_full_detail(src, perms, dist, ba, bl))

    # Per-model summary
    per_model = per_model_mae_summary(full_rows)

    # --- save ---
    out_params  = Path(args.output_params)
    out_summary = Path(args.output_summary)
    out_full    = Path(args.output_full)

    save_parameter_file(out_params, best, global_stats, per_model, perm_summary_rows)
    write_rows_csv(out_summary, perm_summary_rows)
    write_rows_csv(out_full, full_rows)

    # --- terminal ---
    print_terminal_summary(best, global_stats, per_model, list(all_perms.keys()))

    print(f"  Parameter file : {out_params}")
    print(f"  Summary CSV    : {out_summary}")
    print(f"  Full detail CSV: {out_full}\n")


if __name__ == "__main__":
    main()
