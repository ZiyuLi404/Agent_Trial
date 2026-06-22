"""
Run history_borrowing.py for every permutation of bucket-to-model assignments
and save results with names that reflect the order used.

Default output directories:
  diagnosis matrix    → history_borrowing/history_borrowing_result/all_orders/diagnosis/
  conversation matrix → history_borrowing/history_borrowing_result/all_orders/conversation/

File naming:  b<n1>_b<n2>_b<n3>_b<n4>
  where n1..n4 are the bucket numbers assigned to model 1..4 in CSV row order.
  e.g. b2_b1_b3_b4 means model1->bucket2, model2->bucket1, model3->bucket3, model4->bucket4

Usage:
    python history_borrowing/run_all_orders.py
    python history_borrowing/run_all_orders.py \
        --similarity_csv history_borrowing/similarity_matrix/conversation_similarity_matrix.csv
    python history_borrowing/run_all_orders.py --output_dir my/custom/dir
"""

import argparse
import subprocess
import sys
from itertools import permutations
from pathlib import Path


BUCKETS = ["bucket1", "bucket2", "bucket3", "bucket4"]
BASE_OUTPUT_DIR = Path("history_borrowing/history_borrowing_result/all_orders")


def bucket_tag(order: tuple) -> str:
    """'bucket1','bucket3',... -> 'b1_b3_...'"""
    return "_".join(f"b{b.replace('bucket', '')}" for b in order)


def main():
    parser = argparse.ArgumentParser(
        description="Run history_borrowing.py across all 24 bucket-order permutations."
    )
    parser.add_argument(
        "--accuracy_csv",
        default="history_borrowing/accuracy_by_25_cases.csv",
    )
    parser.add_argument(
        "--similarity_csv",
        default="history_borrowing/similarity_matrix/diagnosis_similarity_matrix.csv",
    )
    parser.add_argument(
        "--replicate_map_json",
        default="history_borrowing/replicate_map.json",
    )
    parser.add_argument(
        "--alpha_grid",
        default="0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument(
        "--lambda_grid",
        default="0,5,10,20,50,100,200",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory to save all output files. "
            "Defaults to all_orders/diagnosis/ or all_orders/conversation/ "
            "based on the similarity matrix filename."
        ),
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        sim_stem = Path(args.similarity_csv).stem  # e.g. diagnosis_similarity_matrix
        if "conversation" in sim_stem:
            subfolder = "conversation"
        elif "diagnosis" in sim_stem:
            subfolder = "diagnosis"
        else:
            subfolder = sim_stem
        output_dir = BASE_OUTPUT_DIR / subfolder

    output_dir.mkdir(parents=True, exist_ok=True)

    all_perms = list(permutations(BUCKETS))
    print(f"Running {len(all_perms)} permutations → {output_dir}\n")

    failed = []
    for i, order in enumerate(all_perms, 1):
        tag = bucket_tag(order)
        out_csv  = output_dir / f"results_{tag}.csv"
        out_json = output_dir / f"summary_{tag}.json"
        bucket_order_str = ",".join(order)

        cmd = [
            sys.executable, "history_borrowing/history_borrowing.py",
            "--accuracy_csv",       args.accuracy_csv,
            "--similarity_csv",     args.similarity_csv,
            "--replicate_map_json", args.replicate_map_json,
            "--bucket_order",       bucket_order_str,
            "--output_csv",         str(out_csv),
            "--output_summary_json",str(out_json),
            "--alpha_grid",         args.alpha_grid,
            "--lambda_grid",        args.lambda_grid,
        ]

        print(f"[{i:02d}/24] order={bucket_order_str}  tag={tag}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ERROR:\n{result.stderr.strip()}")
            failed.append(tag)
        else:
            # Pull the one-line MAE summary out of stdout
            for line in result.stdout.splitlines():
                if "MAE" in line or "IMPROVED" in line or "DID NOT" in line:
                    print(f"  {line.strip()}")

    print(f"\n{'='*60}")
    print(f"Done. {len(all_perms) - len(failed)}/{len(all_perms)} succeeded.")
    if failed:
        print(f"Failed orders: {failed}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
