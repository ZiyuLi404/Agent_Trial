#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pairwise LoRA fingerprint runner (Method I, one pair at a time).

Reuses the data loading / tokenizer / model / training machinery from
fingerprint_detector.py and narrows it to a single binary classification
problem: "did this dialogue come from version_a or version_b?".

This module is deliberately thin — it does not reimplement training. It
imports the reusable pieces from fingerprint_detector.py and adds:
    * filtering examples down to exactly two versions
    * an explicit case/run split override (on top of fingerprint_detector's
      batch/random/scenario auto-splits)
    * per-example probability/confidence/margin/entropy scoring
    * the three-file pairwise output contract (metadata / outputs / summary)

See analyze_pairwise_fingerprint.py for running many pairs and aggregating
their outputs.

CLI example:
    python lora_fingerprint/pairwise_fingerprint.py \
        --data_dir results/generate_diagnosis_distribution \
        --version_a deepseek_flash --version_b deepseek_pro \
        --output_dir results/lora_fingerprint_pairwise \
        --text_field full_dialogue --split_mode scenario
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import replace as _replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in (None, ""):  # allow `python lora_fingerprint/pairwise_fingerprint.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, set_seed

from lora_fingerprint.fingerprint_detector import (
    Example,
    TEXT_FIELD_CHOICES,
    TokenizedDataset,
    choose_device,
    family_from_dir,
    load_backbone,
    load_examples,
    maybe_apply_lora,
    split_examples,
    train_model,
)

METRIC_KEYS = [
    "n", "accuracy", "macro_f1", "balanced_accuracy", "auroc",
    "mean_confidence", "mean_p_true", "mean_margin", "mean_entropy",
    "mean_p_a", "mean_p_b",
]


# --------------------------------------------------------------------------- #
# Split resolution
# --------------------------------------------------------------------------- #
RUN_SPLIT_MODES = {"run", "runs", "run_level"}


def parse_int_list(x: list[int] | str | None) -> list[int] | None:
    """Normalize a run/case-id spec: None/""/"all" -> None (no filter); a
    comma string -> list[int]; an already-parsed list passes through."""
    if x is None:
        return None
    if isinstance(x, str):
        if x.strip() == "" or x.strip().lower() == "all":
            return None
        return [int(v.strip()) for v in x.split(",") if v.strip() != ""]
    return list(x)


def _sample_id_of(e: Example) -> str:
    case_id = e.scenario if e.scenario is not None else -1
    run_id = -1 if e.run is None else int(e.run)
    return f"{e.label_name}_case{case_id}_run{run_id}"


def _split_examples_by_run_ids(
    examples: list[Example], train_runs: list[int], test_runs: list[int]
) -> tuple[list[Example], list[Example]]:
    """Manual global run-id split: every case/version uses the same run-id sets."""
    train_run_set = set(train_runs)
    test_run_set = set(test_runs)
    overlap = train_run_set & test_run_set
    if overlap:
        raise ValueError(f"train_runs and test_runs overlap: {sorted(overlap)}")

    train = [e for e in examples if e.run in train_run_set]
    test = [e for e in examples if e.run in test_run_set]
    return train, test


def _split_examples_by_run_fraction(
    examples: list[Example], test_size: float, seed: int
) -> tuple[list[Example], list[Example], dict[str, dict[str, list[int]]]]:
    """Per (case, version) independent random run split: for every case and
    every version separately, shuffle that group's runs and hold out the last
    `test_size` fraction as test. Guarantees every case appears in both train
    and test for every version present (as long as it has >=2 distinct runs).
    """
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    groups: dict[tuple[int | None, str], list[Example]] = defaultdict(list)
    for e in examples:
        groups[(e.scenario, e.label_name)].append(e)

    rng = random.Random(seed)
    train: list[Example] = []
    test: list[Example] = []
    split_assignment: dict[str, dict[str, list[int]]] = {}

    for (case_id, label_name) in sorted(groups.keys(), key=lambda k: (k[0] if k[0] is not None else -1, k[1])):
        group = groups[(case_id, label_name)]
        runs = sorted({e.run for e in group if e.run is not None})
        shuffled_runs = runs[:]
        rng.shuffle(shuffled_runs)

        n_runs = len(shuffled_runs)
        n_test = max(1, int(round(n_runs * test_size)))
        if n_test >= n_runs:
            n_test = n_runs - 1

        test_run_set = set(shuffled_runs[:n_test])
        train_run_set = set(shuffled_runs[n_test:])

        split_assignment[f"case_{case_id}_label_{label_name}"] = {
            "train_runs": sorted(train_run_set),
            "test_runs": sorted(test_run_set),
        }

        for e in group:
            if e.run in test_run_set:
                test.append(e)
            elif e.run in train_run_set:
                train.append(e)

    return train, test, split_assignment


def _resolve_split(
    examples: list[Example],
    split_mode: str,
    seed: int,
    test_size: float,
    train_cases: list[int] | str | None = None,
    test_cases: list[int] | str | None = None,
    train_runs: list[int] | str | None = None,
    test_runs: list[int] | str | None = None,
    split_file: str | None = None,
) -> tuple[list[Example], list[Example], dict[str, Any]]:
    """Decide train/test examples.

    Priority:
      1. split_mode="run" (or "runs"/"run_level") + explicit train_runs/test_runs
         -> manual global run-id split.
      2. split_mode="run" + test_size -> per (case, version) random run-fraction split.
      3. Otherwise: existing case-level behavior — an explicit train_cases/test_cases/
         train_runs/test_runs override (optionally from split_file), else
         fingerprint_detector.split_examples (batch/random/scenario), unchanged.
    """
    if split_file:
        spec = json.loads(Path(split_file).read_text(encoding="utf-8"))
        train_cases = spec.get("train_cases", train_cases)
        test_cases = spec.get("test_cases", test_cases)
        train_runs = spec.get("train_runs", train_runs)
        test_runs = spec.get("test_runs", test_runs)

    if split_mode in RUN_SPLIT_MODES:
        parsed_train_runs = parse_int_list(train_runs)
        parsed_test_runs = parse_int_list(test_runs)

        if parsed_train_runs is not None and parsed_test_runs is not None:
            train, test = _split_examples_by_run_ids(examples, parsed_train_runs, parsed_test_runs)
            info = {
                "split_unit": "run_manual",
                "train_cases": sorted({e.scenario for e in train if e.scenario is not None}),
                "test_cases": sorted({e.scenario for e in test if e.scenario is not None}),
                "train_runs": sorted(set(parsed_train_runs)),
                "test_runs": sorted(set(parsed_test_runs)),
                "test_size": None,
                "split_assignment": None,
            }
            return train, test, info

        train, test, split_assignment = _split_examples_by_run_fraction(examples, test_size, seed)
        info = {
            "split_unit": "run_fraction",
            "train_cases": sorted({e.scenario for e in train if e.scenario is not None}),
            "test_cases": sorted({e.scenario for e in test if e.scenario is not None}),
            "train_runs": None,
            "test_runs": None,
            "test_size": test_size,
            "split_assignment": split_assignment,
        }
        return train, test, info

    has_explicit = any(v is not None for v in (train_cases, test_cases, train_runs, test_runs))
    if has_explicit:
        def _matches(e: Example, cases, runs) -> bool:
            if cases is not None and cases != "all" and e.scenario not in cases:
                return False
            if runs is not None and runs != "all" and e.run not in runs:
                return False
            return True

        train = [e for e in examples if _matches(e, train_cases, train_runs)]
        test = [e for e in examples if _matches(e, test_cases, test_runs)]
        split_unit = "case" if (train_cases is not None or test_cases is not None) else "run"
        info = {
            "split_unit": split_unit,
            "train_cases": sorted(set(train_cases)) if isinstance(train_cases, list) else (train_cases or []),
            "test_cases": sorted(set(test_cases)) if isinstance(test_cases, list) else (test_cases or []),
            "train_runs": train_runs if train_runs is not None else "all",
            "test_runs": test_runs if test_runs is not None else "all",
            "test_size": None,
            "split_assignment": None,
        }
        return train, test, info

    train, test = split_examples(examples, split_mode, test_size, seed)
    split_unit = {"scenario": "case", "batch": "batch", "random": "sample"}.get(split_mode, split_mode)
    info = {
        "split_unit": split_unit,
        "train_cases": sorted({e.scenario for e in train if e.scenario is not None}),
        "test_cases": sorted({e.scenario for e in test if e.scenario is not None}),
        "train_runs": "all",
        "test_runs": "all",
        "test_size": test_size,
        "split_assignment": None,
    }
    return train, test, info


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _predict_rows(
    model: Any,
    examples: list[Example],
    loader: DataLoader,
    device,
    label2id: dict[str, int],
    version_a: str,
    version_b: str,
    split_name: str,
    text_field: str,
) -> list[dict[str, Any]]:
    import torch

    model.eval()
    label_id_a = label2id[version_a]
    label_id_b = label2id[version_b]
    rows: list[dict[str, Any]] = []
    idx = 0
    with torch.no_grad():
        for raw_batch in loader:
            raw_batch = dict(raw_batch)
            raw_batch.pop("labels", None)
            device_batch = {key: value.to(device) for key, value in raw_batch.items()}
            outputs = model(**device_batch)
            logits = outputs.logits.detach().cpu().numpy()
            for i in range(logits.shape[0]):
                e = examples[idx]
                idx += 1
                logit_vec = logits[i]
                probs = _softmax(logit_vec)
                p_a = float(probs[label_id_a])
                p_b = float(probs[label_id_b])
                true_label_id = label2id[e.label_name]
                predicted_label_id = int(np.argmax(probs))
                predicted_version = version_a if predicted_label_id == label_id_a else version_b
                correct = predicted_label_id == true_label_id
                p_true = float(probs[true_label_id])
                p_other = p_b if true_label_id == label_id_a else p_a
                confidence = max(p_a, p_b)
                margin = abs(p_a - p_b)
                entropy = -(p_a * math.log(p_a + 1e-12) + p_b * math.log(p_b + 1e-12))
                run_id = -1 if e.run is None else int(e.run)
                case_id = e.scenario if e.scenario is not None else -1
                rows.append(
                    {
                        "sample_id": f"{e.label_name}_case{case_id}_run{run_id}",
                        "split": split_name,
                        "case_id": case_id,
                        "run_id": run_id,
                        "true_version": e.label_name,
                        "true_label_id": true_label_id,
                        "version_a": version_a,
                        "version_b": version_b,
                        "logit_a": float(logit_vec[label_id_a]),
                        "logit_b": float(logit_vec[label_id_b]),
                        "p_a": p_a,
                        "p_b": p_b,
                        "predicted_version": predicted_version,
                        "predicted_label_id": predicted_label_id,
                        "correct": bool(correct),
                        "p_true": p_true,
                        "p_other": p_other,
                        "confidence": confidence,
                        "margin": margin,
                        "entropy": entropy,
                        "source_file": f"{e.source_dir}/{e.case_file}",
                        "text_field": text_field,
                        "diagnosis_text": e.diagnosis_text,
                        "gold_diagnosis": e.gold_diagnosis,
                        "is_correct_diagnosis": None,
                    }
                )
    return rows


def _metric_block(rows: list[dict[str, Any]], version_b: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {key: (0 if key == "n" else None) for key in METRIC_KEYS}

    y_true = [r["true_label_id"] for r in rows]
    y_pred = [r["predicted_label_id"] for r in rows]
    auc_labels = [1 if r["true_version"] == version_b else 0 for r in rows]
    auc_scores = [r["p_b"] for r in rows]
    auroc = None
    if len(set(auc_labels)) == 2:
        auroc = float(roc_auc_score(auc_labels, auc_scores))

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    return {
        "n": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": auroc,
        "mean_confidence": mean("confidence"),
        "mean_p_true": mean("p_true"),
        "mean_margin": mean("margin"),
        "mean_entropy": mean("entropy"),
        "mean_p_a": mean("p_a"),
        "mean_p_b": mean("p_b"),
    }


def _build_summary(
    rows: list[dict[str, Any]], comparison_id: str, version_a: str, version_b: str, text_field: str
) -> dict[str, Any]:
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    a_rows = [r for r in rows if r["true_version"] == version_a]
    b_rows = [r for r in rows if r["true_version"] == version_b]

    case_ids = sorted({r["case_id"] for r in rows})
    by_case = [
        {"case_id": case_id, **_metric_block([r for r in rows if r["case_id"] == case_id], version_b)}
        for case_id in case_ids
    ]

    return {
        "comparison_id": comparison_id,
        "version_a": version_a,
        "version_b": version_b,
        "text_field": text_field,
        "overall": _metric_block(rows, version_b),
        "by_split": {
            "train": _metric_block(train_rows, version_b),
            "test": _metric_block(test_rows, version_b),
        },
        "by_true_version": {
            version_a: _metric_block(a_rows, version_b),
            version_b: _metric_block(b_rows, version_b),
        },
        "by_case": by_case,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_pairwise_fingerprint(
    data_dir: str,
    version_a: str,
    version_b: str,
    output_dir: str,
    text_field: str = "full_dialogue",
    split_mode: str = "scenario",
    model_name: str = "distilbert-base-uncased",
    use_lora: bool = False,
    load_in_4bit: bool = False,
    max_length: int = 512,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    seed: int = 42,
    train_cases: list[int] | str | None = None,
    test_cases: list[int] | str | None = None,
    train_runs: list[int] | str | None = None,
    test_runs: list[int] | str | None = None,
    split_file: str | None = None,
    overwrite: bool = False,
    # extra knobs, not in the "suggested" signature but needed to preserve
    # full LoRA/QLoRA parity with fingerprint_detector.py's CLI.
    test_size: float = 0.3,
    allow_remote_model_files: bool = False,
    dtype: str = "auto",
    gradient_checkpointing: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "auto",
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
) -> dict[str, Any]:
    """Train + evaluate a binary version_a-vs-version_b fingerprint classifier.

    Writes metadata.json / pairwise_outputs.jsonl / summary_metrics.json under
    <output_dir>/<version_a>__vs__<version_b>/ and returns a dict with paths
    and the summary metrics.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    comparison_id = f"{version_a}__vs__{version_b}"
    pair_dir = output_dir / comparison_id

    metadata_path = pair_dir / "metadata.json"
    outputs_path = pair_dir / "pairwise_outputs.jsonl"
    summary_path = pair_dir / "summary_metrics.json"

    if not overwrite and metadata_path.exists() and outputs_path.exists() and summary_path.exists():
        print(f"[skip] {comparison_id}: outputs already exist at {pair_dir} (overwrite=True to rerun)")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "comparison_id": comparison_id,
            "output_dir": str(pair_dir),
            "metadata_path": str(metadata_path),
            "outputs_path": str(outputs_path),
            "summary_path": str(summary_path),
            "summary": summary,
            "skipped": True,
        }

    set_seed(seed)

    all_examples = load_examples(data_dir, text_field=text_field)

    # version_a/version_b normally name a *family* (e.g. "deepseek_flash",
    # spanning its _1/_2/_3 batch folders). If both instead name exact batch
    # folders (e.g. "deepseek_flash_1" vs "deepseek_flash_2"), compare those
    # two folders directly — useful as a same-model batch-consistency check.
    available_folders = {p.name for p in data_dir.iterdir() if p.is_dir()}
    folder_level = version_a in available_folders and version_b in available_folders

    if folder_level:
        examples = [e for e in all_examples if e.source_dir in (version_a, version_b)]
        examples = [_replace(e, label_name=e.source_dir) for e in examples]
    else:
        examples = [e for e in all_examples if family_from_dir(e.source_dir) in (version_a, version_b)]
    if not examples:
        raise ValueError(f"No examples for versions {version_a!r}/{version_b!r} under {data_dir}")

    label2id = {version_a: 0, version_b: 1}
    id2label = {0: version_a, 1: version_b}

    train_examples, test_examples, split_info = _resolve_split(
        examples,
        split_mode=split_mode,
        seed=seed,
        test_size=test_size,
        train_cases=train_cases,
        test_cases=test_cases,
        train_runs=train_runs,
        test_runs=test_runs,
        split_file=split_file,
    )
    if not train_examples or not test_examples:
        raise ValueError(f"Empty train/test split for {comparison_id} (split_mode={split_mode}).")

    train_ids = {_sample_id_of(e) for e in train_examples}
    test_ids = {_sample_id_of(e) for e in test_examples}
    assert train_ids.isdisjoint(test_ids), f"Train/test overlap detected for {comparison_id}"

    train_args = SimpleNamespace(
        model_name=model_name,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        use_lora=use_lora,
        gradient_checkpointing=gradient_checkpointing,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        logging_steps=logging_steps,
    )

    local_files_only = not allow_remote_model_files
    tokenizer, model = load_backbone(train_args, label2id, id2label, local_files_only)
    model = maybe_apply_lora(model, train_args)

    train_ds = TokenizedDataset(train_examples, label2id, tokenizer, max_length)
    test_ds = TokenizedDataset(test_examples, label2id, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    device = choose_device()
    print(f"\n[{comparison_id}] using device: {device}")
    train_model(model, train_loader, test_loader, train_args, device)

    rows: list[dict[str, Any]] = []
    rows.extend(_predict_rows(model, train_examples, train_eval_loader, device, label2id, version_a, version_b, "train", text_field))
    rows.extend(_predict_rows(model, test_examples, test_loader, device, label2id, version_a, version_b, "test", text_field))

    pair_dir.mkdir(parents=True, exist_ok=True)
    with outputs_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_cases_list = split_info["train_cases"] if isinstance(split_info["train_cases"], list) else []
    test_cases_list = split_info["test_cases"] if isinstance(split_info["test_cases"], list) else []
    metadata = {
        "comparison_id": comparison_id,
        "version_a": version_a,
        "version_b": version_b,
        "task_type": "pairwise_fingerprint_classification",
        "label_unit": "folder" if folder_level else "family",
        "text_field": text_field,
        "split_spec": {
            "split_mode": split_mode,
            "split_unit": split_info["split_unit"],
            "train_cases": train_cases_list,
            "test_cases": test_cases_list,
            "cases": sorted(set(train_cases_list) | set(test_cases_list)),
            "train_runs": split_info["train_runs"],
            "test_runs": split_info["test_runs"],
            "test_size": split_info.get("test_size"),
            "seed": seed,
            "split_assignment": split_info.get("split_assignment"),
            "split_file": split_file,
        },
        "model_config": {
            "backbone": model_name,
            "training_strategy": "qlora" if (use_lora and load_in_4bit) else ("lora" if use_lora else "full_finetune"),
            "use_lora": use_lora,
            "load_in_4bit": load_in_4bit,
            "max_length": max_length,
            "num_labels": 2,
        },
        "training_config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "batch_size": batch_size,
            "seed": seed,
        },
        "label_map": {"0": version_a, "1": version_b},
        "counts": {
            "n_train": len(train_examples),
            "n_test": len(test_examples),
            "n_total": len(train_examples) + len(test_examples),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _build_summary(rows, comparison_id, version_a, version_b, text_field)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{comparison_id}] test acc={summary['by_split']['test']['accuracy']:.4f} "
          f"macro_f1={summary['by_split']['test']['macro_f1']:.4f}")

    return {
        "comparison_id": comparison_id,
        "output_dir": str(pair_dir),
        "metadata_path": str(metadata_path),
        "outputs_path": str(outputs_path),
        "summary_path": str(summary_path),
        "summary": summary,
        "skipped": False,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one pairwise LoRA fingerprint comparison.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--version_a", required=True)
    parser.add_argument("--version_b", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_field", choices=TEXT_FIELD_CHOICES, default="full_dialogue")
    parser.add_argument("--split_mode", choices=["batch", "random", "scenario", "run"], default="scenario")
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--allow_remote_model_files", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--train_cases", default=None, help="comma-separated case ids, or 'all'")
    parser.add_argument("--test_cases", default=None, help="comma-separated case ids, or 'all'")
    parser.add_argument("--train_runs", default=None, help="comma-separated run ids, or 'all'")
    parser.add_argument("--test_runs", default=None, help="comma-separated run ids, or 'all'")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_int_list_or_all(value: str | None) -> list[int] | str | None:
    if value is None:
        return None
    if value.strip().lower() == "all":
        return "all"
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_pairwise_fingerprint(
        data_dir=args.data_dir,
        version_a=args.version_a,
        version_b=args.version_b,
        output_dir=args.output_dir,
        text_field=args.text_field,
        split_mode=args.split_mode,
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
        train_cases=_parse_int_list_or_all(args.train_cases),
        test_cases=_parse_int_list_or_all(args.test_cases),
        train_runs=_parse_int_list_or_all(args.train_runs),
        test_runs=_parse_int_list_or_all(args.test_runs),
        split_file=args.split_file,
        overwrite=args.overwrite,
        test_size=args.test_size,
        allow_remote_model_files=args.allow_remote_model_files,
        dtype=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
    )
    print(f"\nSaved pairwise fingerprint outputs under {result['output_dir']}")


if __name__ == "__main__":
    main()
