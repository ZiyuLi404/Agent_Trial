"""
Analyze all JSON files in a groundtruth directory and output per-bucket accuracy CSV.

Usage:
    python history_borrowing/accuracy_summary.py \
        --groundtruth_dir history_borrowing/groundtruth \
        --bucket_size 25 \
        --output_csv history_borrowing/accuracy_by_25_cases.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def extract_correctness(case: dict, case_index: int, model_name: str):
    """Try common field names/paths to extract a 0/1 correctness value from a case dict."""

    def bool_to_float(v):
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)) and v in (0, 1, 0.0, 1.0):
            return float(v)
        if isinstance(v, str):
            low = v.strip().lower()
            if low in ("true", "yes", "1", "correct"):
                return 1.0
            if low in ("false", "no", "0", "incorrect", "wrong"):
                return 0.0
        return None

    # Flat fields
    for field in ("correct", "is_correct", "correctness", "accuracy", "score"):
        if field in case:
            val = bool_to_float(case[field])
            if val is not None:
                return val

    # Nested paths: result.correct, evaluation.correct, final.correct
    for parent in ("result", "evaluation", "final"):
        if parent in case and isinstance(case[parent], dict):
            for field in ("correct", "is_correct", "correctness", "accuracy", "score"):
                if field in case[parent]:
                    val = bool_to_float(case[parent][field])
                    if val is not None:
                        return val

    print(
        f"  WARNING [{model_name}] case index {case_index}: "
        "no detectable correctness field — skipping.",
        file=sys.stderr,
    )
    return None


def load_cases(json_path: Path):
    """Load a JSON file and return a list of 0/1 correctness values."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    model_name = json_path.stem

    # Find the cases list — try common container keys, then fall back to top-level list.
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        found = False
        for key in ("results", "cases", "data", "items", "records"):
            if key in data and isinstance(data[key], list):
                cases = data[key]
                found = True
                break
        if not found:
            print(
                f"  WARNING [{model_name}]: cannot find a list of cases in JSON — "
                "no recognised container key (results/cases/data/items/records).",
                file=sys.stderr,
            )
            return []
    else:
        print(
            f"  WARNING [{model_name}]: unexpected JSON root type {type(data)} — skipping.",
            file=sys.stderr,
        )
        return []

    scores = []
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            print(
                f"  WARNING [{model_name}] case index {i}: not a dict — skipping.",
                file=sys.stderr,
            )
            continue
        val = extract_correctness(case, i, model_name)
        if val is not None:
            scores.append(val)

    return scores


def compute_accuracies(scores, bucket_size, model_name) -> dict:
    """Return total accuracy and per-bucket accuracies as a plain dict."""
    n = len(scores)
    if n == 0:
        return {"total": None}

    total = sum(scores) / n
    result: dict = {"total": round(total, 4)}

    # Split into complete buckets only; warn about remainder.
    bucket_idx = 1
    start = 0
    while start + bucket_size <= n:
        bucket = scores[start : start + bucket_size]
        result[f"bucket{bucket_idx}"] = round(sum(bucket) / len(bucket), 4)
        start += bucket_size
        bucket_idx += 1

    remainder = n - start
    if remainder > 0:
        print(
            f"  NOTE [{model_name}]: {remainder} trailing case(s) did not fill a complete "
            f"bucket (bucket_size={bucket_size}) and were excluded from bucket columns.",
            file=sys.stderr,
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-bucket model accuracy from groundtruth JSON files."
    )
    parser.add_argument(
        "--groundtruth_dir",
        default="history_borrowing/groundtruth",
        help="Directory containing .json groundtruth files.",
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=25,
        help="Number of cases per bucket (default: 25).",
    )
    parser.add_argument(
        "--output_csv",
        default="history_borrowing/accuracy_by_25_cases.csv",
        help="Path for the output CSV file.",
    )
    args = parser.parse_args()

    groundtruth_dir = Path(args.groundtruth_dir)
    if not groundtruth_dir.is_dir():
        sys.exit(f"ERROR: groundtruth_dir does not exist: {groundtruth_dir}")

    json_files = sorted(groundtruth_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"ERROR: no .json files found in {groundtruth_dir}")

    rows = []
    all_bucket_keys = set()

    print(f"\nProcessing {len(json_files)} file(s) from: {groundtruth_dir}\n")

    for json_path in json_files:
        model_name = json_path.stem
        print(f"  {model_name} ...")
        scores = load_cases(json_path)
        if not scores:
            print(
                f"  WARNING [{model_name}]: no valid cases — skipping model.",
                file=sys.stderr,
            )
            continue

        acc = compute_accuracies(scores, args.bucket_size, model_name)
        acc["model"] = model_name
        rows.append(acc)
        all_bucket_keys.update(k for k in acc if k.startswith("bucket"))

    if not rows:
        sys.exit("ERROR: no models with valid data found.")

    bucket_cols = sorted(all_bucket_keys, key=lambda x: int(x.replace("bucket", "")))
    fieldnames = ["model", "total"] + bucket_cols

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Terminal summary
    col_w = 10
    header = f"{'Model':<28} {'Total':>{col_w}}"
    for bc in bucket_cols:
        header += f"  {bc:>{col_w}}"
    print(f"\n{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['model']:<28} {str(row.get('total', 'N/A')):>{col_w}}"
        for bc in bucket_cols:
            val = str(row.get(bc, ""))
            line += f"  {val:>{col_w}}"
        print(line)
    print(f"{'='*len(header)}")
    print(f"\nCSV saved to: {output_path}\n")


if __name__ == "__main__":
    main()
