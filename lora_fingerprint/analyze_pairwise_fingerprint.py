#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run many pairwise LoRA fingerprint comparisons and aggregate their outputs.

This script contains no training logic of its own — it discovers/decides
which version pairs to compare, calls
lora_fingerprint.pairwise_fingerprint.run_pairwise_fingerprint for each pair,
and aggregates the per-pair outputs into analysis-wide CSV/JSON files.

CLI examples:
    # all 10 pairs among 5 versions
    python lora_fingerprint/analyze_pairwise_fingerprint.py \
        --data_dir results/generate_diagnosis_distribution \
        --output_dir results/lora_fingerprint_pairwise \
        --analysis_dir results/lora_fingerprint_pairwise_analysis \
        --versions deepseek_flash,deepseek_pro,gpt_5_4_mini,gpt_5_5,qwen_plus \
        --text_field full_dialogue --split_mode scenario --skip_existing

    # only two specific pairs
    python lora_fingerprint/analyze_pairwise_fingerprint.py \
        --data_dir results/generate_diagnosis_distribution \
        --output_dir results/lora_fingerprint_pairwise \
        --analysis_dir results/lora_fingerprint_pairwise_analysis \
        --pairs deepseek_flash:deepseek_pro,gpt_5_4_mini:gpt_5_5 \
        --text_field full_dialogue --split_mode scenario
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python lora_fingerprint/analyze_pairwise_fingerprint.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lora_fingerprint.fingerprint_detector import TEXT_FIELD_CHOICES, family_from_dir
from lora_fingerprint.pairwise_fingerprint import parse_int_list, run_pairwise_fingerprint

ALL_PAIR_SUMMARY_COLUMNS = [
    "comparison_id", "version_a", "version_b", "n_train", "n_test",
    "test_accuracy", "test_macro_f1", "test_balanced_accuracy", "test_auroc",
    "test_mean_confidence", "test_mean_p_true", "test_mean_margin", "test_mean_entropy",
    "test_mean_p_a", "test_mean_p_b",
    "overall_accuracy", "overall_macro_f1", "overall_auroc",
]

DIALOG_FEATURE_COLUMNS = [
    "comparison_id", "split", "sample_id", "case_id", "run_id",
    "true_version", "version_a", "version_b", "predicted_version", "correct",
    "logit_a", "logit_b", "logit_diff_ab", "abs_logit_diff",
    "p_a", "p_b", "p_true", "p_other", "confidence", "margin", "entropy",
    "source_file", "diagnosis_text", "gold_diagnosis", "is_correct_diagnosis",
]

BORROWING_BASE_COLUMNS = [
    "comparison_id", "split", "sample_id", "case_id", "run_id",
    "source_version", "version_a", "version_b", "target_version",
    "p_source_version", "p_other_version", "p_target_version",
    "fingerprint_distance_prob", "fingerprint_similarity_prob", "logit_distance",
]


# --------------------------------------------------------------------------- #
# Pair discovery / selection
# --------------------------------------------------------------------------- #
def discover_versions(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        print(f"[WARN] data_dir {data_dir} does not exist; cannot discover versions.")
        return []
    return sorted({family_from_dir(p.name) for p in data_dir.iterdir() if p.is_dir()})


def build_same_version_pairs(data_dir: Path) -> list[tuple[str, str]]:
    """All batch-vs-batch pairs within each version family, e.g. deepseek_flash_1
    vs deepseek_flash_2 (and _1 vs _3, _2 vs _3 if a third batch exists).
    Useful as a same-model consistency check, distinct from cross-version pairs.
    """
    if not data_dir.exists():
        return []
    folders = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    by_family: dict[str, list[str]] = {}
    for folder in folders:
        by_family.setdefault(family_from_dir(folder), []).append(folder)

    pairs: list[tuple[str, str]] = []
    for _family, batch_folders in sorted(by_family.items()):
        if len(batch_folders) >= 2:
            pairs.extend(itertools.combinations(sorted(batch_folders), 2))
    return pairs


def build_pairs(versions: list[str] | None, pairs_arg: str | None, data_dir: Path) -> list[tuple[str, str]]:
    discovered = discover_versions(data_dir)

    if pairs_arg:
        pairs: list[tuple[str, str]] = []
        for chunk in pairs_arg.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            a, sep, b = chunk.partition(":")
            if not sep:
                raise ValueError(f"Bad --pairs entry {chunk!r}, expected 'version_a:version_b'")
            a, b = a.strip(), b.strip()
            for v in (a, b):
                if discovered and v not in discovered:
                    print(f"[WARN] version {v!r} not found under {data_dir} (discovered: {discovered})")
            pairs.append((a, b))
        return pairs

    if versions:
        for v in versions:
            if discovered and v not in discovered:
                print(f"[WARN] version {v!r} not found under {data_dir} (discovered: {discovered})")
        use_versions = versions
    else:
        use_versions = discovered
        print(f"[info] no --versions/--pairs given; using discovered versions: {discovered}")

    if len(use_versions) < 2:
        raise ValueError(f"Need at least 2 versions to compare, got {use_versions}")
    return list(itertools.combinations(use_versions, 2))


# --------------------------------------------------------------------------- #
# Running pairs
# --------------------------------------------------------------------------- #
def run_all_pairs(
    pairs: list[tuple[str, str]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    output_dir = Path(args.output_dir)
    for version_a, version_b in pairs:
        comparison_id = f"{version_a}__vs__{version_b}"
        pair_dir = output_dir / comparison_id
        metadata_path = pair_dir / "metadata.json"
        outputs_path = pair_dir / "pairwise_outputs.jsonl"
        summary_path = pair_dir / "summary_metrics.json"
        already_exists = all(p.exists() for p in (metadata_path, outputs_path, summary_path))

        if args.skip_existing and already_exists and not args.overwrite:
            print(f"[skip_existing] {comparison_id}")
            # still usable for aggregation — just read what's already on disk instead of retraining.
            skipped.append(
                {
                    "comparison_id": comparison_id,
                    "version_a": version_a,
                    "version_b": version_b,
                    "output_dir": str(pair_dir),
                    "metadata_path": str(metadata_path),
                    "outputs_path": str(outputs_path),
                    "summary_path": str(summary_path),
                    "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                }
            )
            continue

        try:
            result = run_pairwise_fingerprint(
                data_dir=args.data_dir,
                version_a=version_a,
                version_b=version_b,
                output_dir=args.output_dir,
                text_field=args.text_field,
                split_mode=args.split_mode,
                test_size=args.test_size,
                model_name=args.model_name,
                use_lora=args.use_lora,
                load_in_4bit=args.load_in_4bit,
                max_length=args.max_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                seed=args.seed,
                train_runs=parse_int_list(args.train_runs),
                test_runs=parse_int_list(args.test_runs),
                overwrite=args.overwrite,
                allow_remote_model_files=args.allow_remote_model_files,
            )
            completed.append(result)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED] {comparison_id}: {exc}")
            traceback.print_exc()
            failed.append(
                {"comparison_id": comparison_id, "version_a": version_a, "version_b": version_b, "error": str(exc)}
            )
            if args.fail_fast:
                raise

    return completed, skipped, failed


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def write_all_pair_summary(completed: list[dict[str, Any]], analysis_dir: Path) -> None:
    rows = []
    for result in completed:
        summary = result["summary"]
        train = summary["by_split"]["train"]
        test = summary["by_split"]["test"]
        overall = summary["overall"]
        rows.append(
            {
                "comparison_id": summary["comparison_id"],
                "version_a": summary["version_a"],
                "version_b": summary["version_b"],
                "n_train": train["n"],
                "n_test": test["n"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_auroc": test["auroc"],
                "test_mean_confidence": test["mean_confidence"],
                "test_mean_p_true": test["mean_p_true"],
                "test_mean_margin": test["mean_margin"],
                "test_mean_entropy": test["mean_entropy"],
                "test_mean_p_a": test["mean_p_a"],
                "test_mean_p_b": test["mean_p_b"],
                "overall_accuracy": overall["accuracy"],
                "overall_macro_f1": overall["macro_f1"],
                "overall_auroc": overall["auroc"],
            }
        )

    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "all_pair_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (analysis_dir / "all_pair_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_PAIR_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _iter_pair_rows(completed: list[dict[str, Any]]):
    for result in completed:
        comparison_id = result["comparison_id"]
        outputs_path = Path(result["outputs_path"])
        with outputs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["comparison_id"] = comparison_id
                yield row


def write_all_dialog_features(completed: list[dict[str, Any]], analysis_dir: Path) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "all_dialog_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIALOG_FEATURE_COLUMNS)
        writer.writeheader()
        for row in _iter_pair_rows(completed):
            logit_diff_ab = row["logit_a"] - row["logit_b"]
            writer.writerow(
                {
                    "comparison_id": row["comparison_id"],
                    "split": row["split"],
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "run_id": row["run_id"],
                    "true_version": row["true_version"],
                    "version_a": row["version_a"],
                    "version_b": row["version_b"],
                    "predicted_version": row["predicted_version"],
                    "correct": row["correct"],
                    "logit_a": row["logit_a"],
                    "logit_b": row["logit_b"],
                    "logit_diff_ab": logit_diff_ab,
                    "abs_logit_diff": abs(logit_diff_ab),
                    "p_a": row["p_a"],
                    "p_b": row["p_b"],
                    "p_true": row["p_true"],
                    "p_other": row["p_other"],
                    "confidence": row["confidence"],
                    "margin": row["margin"],
                    "entropy": row["entropy"],
                    "source_file": row["source_file"],
                    "diagnosis_text": row["diagnosis_text"],
                    "gold_diagnosis": row["gold_diagnosis"],
                    "is_correct_diagnosis": row["is_correct_diagnosis"],
                }
            )


def _format_lambda(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def write_borrowing_features(
    completed: list[dict[str, Any]],
    analysis_dir: Path,
    target_version: str | None,
    lambda_values: list[float],
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    prob_cols = [f"exp_weight_prob_lambda_{_format_lambda(v)}" for v in lambda_values]
    logit_cols = [f"exp_weight_logit_lambda_{_format_lambda(v)}" for v in lambda_values]
    columns = BORROWING_BASE_COLUMNS + prob_cols + logit_cols

    with (analysis_dir / "borrowing_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in _iter_pair_rows(completed):
            version_a = row["version_a"]
            version_b = row["version_b"]
            fingerprint_distance_prob = abs(row["p_a"] - row["p_b"])
            fingerprint_similarity_prob = 1 - fingerprint_distance_prob
            logit_distance = abs(row["logit_a"] - row["logit_b"])

            p_target_version = None
            if target_version == version_a:
                p_target_version = row["p_a"]
            elif target_version == version_b:
                p_target_version = row["p_b"]

            out = {
                "comparison_id": row["comparison_id"],
                "split": row["split"],
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "run_id": row["run_id"],
                "source_version": row["true_version"],
                "version_a": version_a,
                "version_b": version_b,
                "target_version": target_version,
                "p_source_version": row["p_true"],
                "p_other_version": row["p_other"],
                "p_target_version": p_target_version,
                "fingerprint_distance_prob": fingerprint_distance_prob,
                "fingerprint_similarity_prob": fingerprint_similarity_prob,
                "logit_distance": logit_distance,
            }
            for lam, col in zip(lambda_values, prob_cols):
                out[col] = math.exp(-lam * fingerprint_distance_prob)
            for lam, col in zip(lambda_values, logit_cols):
                out[col] = math.exp(-lam * logit_distance)
            writer.writerow(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run many pairwise LoRA fingerprint comparisons and aggregate results."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True, help="per-pair outputs are written under here")
    parser.add_argument("--analysis_dir", required=True, help="aggregated analysis files are written here")
    parser.add_argument("--versions", default=None, help="comma-separated version names")
    parser.add_argument("--pairs", default=None, help="comma-separated explicit pairs, e.g. A:B,C:D")
    parser.add_argument("--include_same_version_batches", action="store_true",
                         help="also run batch-vs-batch pairs within each version, "
                              "e.g. deepseek_flash_1 vs deepseek_flash_2")
    parser.add_argument("--text_field", choices=TEXT_FIELD_CHOICES, default="full_dialogue")
    parser.add_argument("--split_mode", choices=["batch", "random", "scenario", "run"], default="scenario")
    parser.add_argument("--test_size", type=float, default=0.3,
                         help="fraction held out for split_mode scenario/random/run")
    parser.add_argument("--train_runs", default=None,
                         help="manual split_mode=run override: comma-separated run ids, or 'all'")
    parser.add_argument("--test_runs", default=None,
                         help="manual split_mode=run override: comma-separated run ids, or 'all'")
    parser.add_argument("--skip_existing", action="store_true", help="skip a pair if its outputs already exist")
    parser.add_argument("--overwrite", action="store_true", help="rerun and overwrite existing pair outputs")
    parser.add_argument("--fail_fast", action="store_true", help="stop immediately if a pair fails")
    parser.add_argument("--target_version", default=None, help="version to compute borrowing features against")
    parser.add_argument("--lambda_values", default="1,5,10,50,100")
    # pass-through training args (see pairwise_fingerprint.run_pairwise_fingerprint)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--allow_remote_model_files", action="store_true",
                         help="allow downloading the backbone from Hugging Face if not cached locally")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    data_dir = Path(args.data_dir)
    analysis_dir = Path(args.analysis_dir)

    versions = [v.strip() for v in args.versions.split(",") if v.strip()] if args.versions else None
    pairs = build_pairs(versions, args.pairs, data_dir)

    if args.include_same_version_batches:
        same_version_pairs = build_same_version_pairs(data_dir)
        print(f"[info] --include_same_version_batches: adding {len(same_version_pairs)} same-version batch pair(s)")
        pairs = pairs + same_version_pairs

    print(f"Running {len(pairs)} pairwise comparison(s):")
    for a, b in pairs:
        print(f"  {a} vs {b}")

    completed, skipped, failed = run_all_pairs(pairs, args)

    lambda_values = [float(v.strip()) for v in args.lambda_values.split(",") if v.strip()]

    # Aggregation covers both freshly-trained and skip_existing pairs — both
    # have valid metadata/outputs/summary files on disk, and dropping the
    # skipped ones from the CSVs would defeat the point of --skip_existing.
    aggregatable = completed + skipped
    if aggregatable:
        write_all_pair_summary(aggregatable, analysis_dir)
        write_all_dialog_features(aggregatable, analysis_dir)
        write_borrowing_features(aggregatable, analysis_dir, args.target_version, lambda_values)
    else:
        print("[WARN] no pairs available (completed or skipped); skipping aggregation outputs.")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_metadata = {
        "data_dir": str(data_dir),
        "output_dir": str(Path(args.output_dir)),
        "analysis_dir": str(analysis_dir),
        "text_field": args.text_field,
        "split_mode": args.split_mode,
        "test_size": args.test_size,
        "model_name": args.model_name,
        "use_lora": args.use_lora,
        "load_in_4bit": args.load_in_4bit,
        "target_version": args.target_version,
        "lambda_values": lambda_values,
        "pairs_requested": [f"{a}__vs__{b}" for a, b in pairs],
        "completed": [c["comparison_id"] for c in completed],
        "skipped": [s["comparison_id"] for s in skipped],
        "failed": [f["comparison_id"] for f in failed],
    }
    (analysis_dir / "analysis_metadata.json").write_text(
        json.dumps(analysis_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Pairwise fingerprint run summary ===")
    print(f"completed: {len(completed)}")
    for c in completed:
        print(f"  - {c['comparison_id']}")
    print(f"skipped:   {len(skipped)}")
    for s in skipped:
        print(f"  - {s['comparison_id']}")
    print(f"failed:    {len(failed)}")
    for fr in failed:
        print(f"  - {fr['comparison_id']}: {fr['error']}")
    print(f"\nAggregated analysis written to {analysis_dir}")


if __name__ == "__main__":
    main()
