#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA model-fingerprint detector (模型指纹识别 · Method I).

给一段问诊文本，判断它出自哪个 LLM（gpt_5_5 / deepseek_pro / Qwen_plus_turbo ...）。
这是对 FDLLM (arXiv:2501.16029) 思路的轻量落地：用一个 encoder 文本分类器
（可选挂 LoRA）去学习「不同模型的文风指纹」，在本仓库的问诊对话数据上评测。

与 embedding_similarity / kl_js_divergence 完全并列、互不 import，只读共享数据目录：
    <data_dir>/deepseek_flash_1/case_2.json
    <data_dir>/gpt_5_5_2/case_5.json
每个 case_*.json 是 generate_diagnosis_distribution 的产出（含 samples / full_dialogue / diagnosis_text）。

模型家族（= 分类标签）从文件夹名自动推断（去掉尾部批次号 _<digits>），无需硬编码：
    deepseek_flash_1 / deepseek_flash_2 -> deepseek_flash
    gpt_5_4_mini_1   / gpt_5_4_mini_2   -> gpt_5_4_mini

两个并列实验由 --text_field 决定（对齐 embedding_similarity 的设计）：
    --text_field full_dialogue   学整段问诊对话的指纹（过程层面，默认）
    --text_field diagnosis_text  只学最终诊断那句话（结论层面）

训练/测试划分 --split_mode：
    batch （默认）  每个家族 _1 批 -> 训练， _2 批 -> 测试（留出一整批，诚实的泛化评测）
    random          按标签分层随机划分（--test_size 控制比例）
    scenario        按病例编号留出（跨批次），测「在没见过的病例上还能不能认出版本」

Pairwise output mode (two-version comparisons):
    When exactly two labels are present (or --version_a/--version_b select two), the script
    writes to results/lora_fingerprint_pairwise/<comparison_id>/ with three main files:
        metadata.json         — experiment setup
        pairwise_outputs.jsonl — per-sample logits/probs/scores
        summary_metrics.json  — descriptive aggregate metrics
    plus best_model/ and label_map.json for model reload.

依赖：torch / transformers / scikit-learn / numpy（仓库已装）。
      --use_lora 时额外需要 `peft`（pip install peft），不开则完全不需要。

示例：
    python lora_fingerprint/fingerprint_detector.py \
        --text_field full_dialogue --epochs 5 --batch_size 4
    python lora_fingerprint/fingerprint_detector.py \
        --text_field diagnosis_text --use_lora --lora_r 16
    python lora_fingerprint/fingerprint_detector.py \
        --version_a gpt_5_5 --version_b deepseek_flash --text_field full_dialogue
    python lora_fingerprint/fingerprint_detector.py --prepare_only
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

BATCH_SUFFIX_RE = re.compile(r"_\d+$")
TEXT_FIELD_CHOICES = ["full_dialogue", "diagnosis_text"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Example:
    text: str
    label_name: str
    source_dir: str
    case_file: str
    run: int | None
    scenario: int | None = None
    diagnosis_text: str | None = None
    gold_diagnosis: str | None = None
    is_correct_diagnosis: bool | None = None


def scenario_of(case_file: str) -> int | None:
    m = re.search(r"case_(\d+)", case_file)
    return int(m.group(1)) if m else None


def family_from_dir(dirname: str) -> str:
    return BATCH_SUFFIX_RE.sub("", dirname)


def batch_from_dir(dirname: str) -> str | None:
    m = re.search(r"_(\d+)$", dirname)
    return m.group(1) if m else None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return "\n".join(str(value).replace("\r", "\n").splitlines()).strip()


def _parse_gold_diagnosis(case_obj: dict) -> str | None:
    for key in ("gold_diagnosis", "disease", "diagnosis", "correct_diagnosis", "label"):
        v = case_obj.get(key)
        if v is not None:
            return str(v).strip() or None
    return None


def load_examples(data_dir: Path, text_field: str) -> list[Example]:
    examples: list[Example] = []
    for model_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label_name = family_from_dir(model_dir.name)
        for case_path in sorted(model_dir.glob("case_*.json")):
            try:
                case_obj = json.loads(case_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] failed to read {case_path}: {exc}")
                continue
            samples = case_obj.get("samples", [])
            if not isinstance(samples, list):
                continue
            gold_diag = _parse_gold_diagnosis(case_obj)
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                text = normalize_text(sample.get(text_field))
                if not text:
                    continue
                diag_text = sample.get("diagnosis_text")
                if diag_text is not None:
                    diag_text = str(diag_text).strip() or None

                is_correct: bool | None = None
                if "is_correct" in sample:
                    is_correct = bool(sample["is_correct"])
                elif diag_text is not None and gold_diag is not None:
                    is_correct = diag_text.strip().lower() == gold_diag.strip().lower()

                examples.append(
                    Example(
                        text=text,
                        label_name=label_name,
                        source_dir=model_dir.name,
                        case_file=case_path.name,
                        run=sample.get("run"),
                        scenario=scenario_of(case_path.name),
                        diagnosis_text=diag_text,
                        gold_diagnosis=gold_diag,
                        is_correct_diagnosis=is_correct,
                    )
                )
    return examples


def split_examples(
    examples: list[Example], mode: str, test_size: float, seed: int
) -> tuple[list[Example], list[Example]]:
    if mode == "batch":
        train = [e for e in examples if batch_from_dir(e.source_dir) == "1"]
        test = [e for e in examples if batch_from_dir(e.source_dir) == "2"]
        unassigned = [e for e in examples if batch_from_dir(e.source_dir) not in {"1", "2"}]
        if unassigned:
            print(
                f"[WARN] {len(unassigned)} examples are not in a _1/_2 batch and were dropped "
                f"under split_mode=batch. Use --split_mode random to keep them."
            )
        return train, test

    if mode == "scenario":
        scenarios = sorted({e.scenario for e in examples if e.scenario is not None})
        if len(scenarios) < 2:
            raise ValueError("split_mode=scenario needs at least 2 distinct scenarios.")
        rng = random.Random(seed)
        shuffled = scenarios[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_size)))
        test_scenarios = set(shuffled[:n_test])
        train_scenarios = set(shuffled[n_test:])
        print(f"[scenario split] train scenarios={sorted(train_scenarios)}  "
              f"test scenarios={sorted(test_scenarios)}")
        train = [e for e in examples if e.scenario in train_scenarios]
        test = [e for e in examples if e.scenario in test_scenarios]
        return train, test

    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = {}
    for e in examples:
        by_label.setdefault(e.label_name, []).append(e)
    train, test = [], []
    for label, items in sorted(by_label.items()):
        items = items[:]
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_size)))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


def build_label_maps(examples: list[Example]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({e.label_name for e in examples})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def print_split_summary(train: list[Example], test: list[Example]) -> None:
    print("Split summary:")
    print(f"  train: {len(train)}")
    print(f"  test:  {len(test)}")
    print("\nLabel summary:")
    for name, items in (("train", train), ("test", test)):
        counts = Counter(e.label_name for e in items)
        print(f"  {name}:")
        for label, count in sorted(counts.items()):
            print(f"    {label}: {count}")


def save_dataset_preview(train: list[Example], test: list[Example], label2id: dict[str, int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in (("train", train), ("test", test)):
        path = out_dir / f"{name}_examples.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e in items:
                f.write(
                    json.dumps(
                        {
                            "text": e.text,
                            "label": label2id[e.label_name],
                            "label_name": e.label_name,
                            "source_dir": e.source_dir,
                            "case_file": e.case_file,
                            "run": -1 if e.run is None else int(e.run),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
class TokenizedDataset(Dataset):
    def __init__(self, examples: list[Example], label2id: dict[str, int], tokenizer: Any, max_length: int):
        self.labels = [label2id[e.label_name] for e in examples]
        self.encodings = tokenizer(
            [e.text for e in examples],
            truncation=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: values[idx] for key, values in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
LORA_TARGETS_BY_TYPE = {
    "distilbert": ["q_lin", "v_lin"],
    "bert": ["query", "value"],
    "qwen2": ["q_proj", "v_proj"],
    "qwen3": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
}


def resolve_lora_targets(model: Any, spec: str) -> list[str] | None:
    if spec and spec != "auto":
        return [m.strip() for m in spec.split(",") if m.strip()]
    model_type = getattr(model.config, "model_type", "")
    return LORA_TARGETS_BY_TYPE.get(model_type)


def maybe_apply_lora(model: Any, args: argparse.Namespace) -> Any:
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "You passed --use_lora but `peft` is not installed. "
            "Run `pip install peft`, or drop --use_lora to fine-tune the full model."
        ) from exc

    if getattr(args, "load_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=resolve_lora_targets(model, args.lora_target_modules),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def load_backbone(args: argparse.Namespace, label2id: dict, id2label: dict, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"auto": "auto", "float16": torch.float16,
             "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    load_kwargs: dict[str, Any] = dict(
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        local_files_only=local_files_only,
    )
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype

    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load_in_4bit (QLoRA) needs a CUDA GPU + bitsandbytes; not available here.")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **load_kwargs)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing and not args.load_in_4bit:
        model.gradient_checkpointing_enable()
    return tokenizer, model


def _is_dispatched(model: Any) -> bool:
    return getattr(model, "hf_device_map", None) is not None or getattr(model, "is_loaded_in_4bit", False)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(model: Any, dataloader: DataLoader, device: torch.device):
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            predictions = torch.argmax(outputs.logits, dim=-1)
            y_true.extend(batch["labels"].detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
    }
    return metrics, y_true_arr, y_pred_arr


def train_model(model, train_loader, test_loader, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, int(len(train_loader) * args.epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    whole_epochs = int(args.epochs)

    if not _is_dispatched(model):
        model.to(device)
    for epoch in range(whole_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print(f"step {global_step}: train_loss={np.mean(epoch_losses):.4f}")

        metrics, _, _ = evaluate_model(model, test_loader, device)
        print(
            f"epoch {epoch + 1}/{whole_epochs}: "
            f"train_loss={np.mean(epoch_losses):.4f} "
            f"eval_loss={metrics['loss']:.4f} "
            f"eval_acc={metrics['accuracy']:.4f} "
            f"eval_macro_f1={metrics['macro_f1']:.4f}"
        )
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate_model(model, test_loader, device)


# --------------------------------------------------------------------------- #
# Pairwise evaluation — per-sample records
# --------------------------------------------------------------------------- #
def collect_pairwise_records(
    model: Any,
    examples: list[Example],
    dataloader: DataLoader,
    device: torch.device,
    label_id_a: int,
    label_id_b: int,
    split_name: str,
    version_a: str,
    version_b: str,
    data_dir: Path,
    text_field: str,
    save_text: bool = False,
) -> list[dict]:
    """One forward pass; return one dict per sample with logits, probs, and metadata."""
    model.eval()
    records: list[dict] = []
    example_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            true_labels = batch["labels"].detach().cpu().tolist()
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_gpu)

            logits = outputs.logits.detach().cpu()   # (B, num_labels)
            probs = torch.softmax(logits, dim=-1)    # (B, num_labels)

            for i in range(len(true_labels)):
                e = examples[example_idx]
                example_idx += 1

                la = float(logits[i, label_id_a])
                lb = float(logits[i, label_id_b])
                pa = float(probs[i, label_id_a])
                pb = float(probs[i, label_id_b])

                true_label_id: int = true_labels[i]
                true_version = version_a if true_label_id == label_id_a else version_b

                pred_label_id = label_id_a if pa >= pb else label_id_b
                predicted_version = version_a if pred_label_id == label_id_a else version_b

                correct = pred_label_id == true_label_id
                p_true = pa if true_label_id == label_id_a else pb
                p_other = pb if true_label_id == label_id_a else pa
                confidence = max(pa, pb)
                margin = abs(pa - pb)
                ent = -(pa * math.log(pa + 1e-12) + pb * math.log(pb + 1e-12))

                case_id = e.scenario
                run_id = int(e.run) if e.run is not None else None
                case_str = str(case_id) if case_id is not None else "unknown"
                run_str = str(run_id) if run_id is not None else "unknown"
                sample_id = f"{true_version}_case{case_str}_run{run_str}"

                source_file = str(data_dir / e.source_dir / e.case_file)

                record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "split": split_name,
                    "case_id": case_id,
                    "run_id": run_id,
                    "true_version": true_version,
                    "true_label_id": true_label_id,
                    "version_a": version_a,
                    "version_b": version_b,
                    "logit_a": round(la, 6),
                    "logit_b": round(lb, 6),
                    "p_a": round(pa, 6),
                    "p_b": round(pb, 6),
                    "predicted_version": predicted_version,
                    "predicted_label_id": pred_label_id,
                    "correct": correct,
                    "p_true": round(p_true, 6),
                    "p_other": round(p_other, 6),
                    "confidence": round(confidence, 6),
                    "margin": round(margin, 6),
                    "entropy": round(ent, 6),
                    "source_file": source_file,
                    "text_field": text_field,
                    "diagnosis_text": e.diagnosis_text,
                    "gold_diagnosis": e.gold_diagnosis,
                    "is_correct_diagnosis": e.is_correct_diagnosis,
                }
                if save_text:
                    record["text"] = e.text

                records.append(record)

    return records


# --------------------------------------------------------------------------- #
# Pairwise output helpers
# --------------------------------------------------------------------------- #
def _safe_round(v: float | None, ndigits: int = 6) -> float | None:
    return round(v, ndigits) if v is not None else None


def _subset_metrics(records: list[dict], version_b: str) -> dict:
    """Aggregate metrics for an arbitrary subset of pairwise records."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    y_true = [1 if r["true_version"] == version_b else 0 for r in records]
    y_pred = [1 if r["predicted_version"] == version_b else 0 for r in records]
    scores = [r["p_b"] for r in records]

    accuracy = float(np.mean([r["correct"] for r in records]))

    try:
        macro_f1: float | None = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        macro_f1 = None

    try:
        bal_acc: float | None = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal_acc = None

    auroc: float | None = None
    if len(set(y_true)) >= 2:
        try:
            auroc = float(roc_auc_score(y_true, scores))
        except Exception:
            auroc = None

    return {
        "n": n,
        "accuracy": round(accuracy, 6),
        "macro_f1": _safe_round(macro_f1),
        "balanced_accuracy": _safe_round(bal_acc),
        "auroc": _safe_round(auroc),
        "mean_confidence": round(float(np.mean([r["confidence"] for r in records])), 6),
        "mean_p_true": round(float(np.mean([r["p_true"] for r in records])), 6),
        "mean_margin": round(float(np.mean([r["margin"] for r in records])), 6),
        "mean_entropy": round(float(np.mean([r["entropy"] for r in records])), 6),
        "mean_p_a": round(float(np.mean([r["p_a"] for r in records])), 6),
        "mean_p_b": round(float(np.mean([r["p_b"] for r in records])), 6),
    }


def _by_version_metrics(all_records: list[dict], version_a: str, version_b: str) -> dict:
    y_true = [1 if r["true_version"] == version_b else 0 for r in all_records]
    y_pred = [1 if r["predicted_version"] == version_b else 0 for r in all_records]

    # label 0 = version_a, label 1 = version_b
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    result: dict[str, dict] = {}
    for version, label_idx in [(version_a, 0), (version_b, 1)]:
        subset = [r for r in all_records if r["true_version"] == version]
        if not subset:
            continue
        n = len(subset)
        acc = float(np.mean([r["correct"] for r in subset]))
        result[version] = {
            "n": n,
            "accuracy": round(acc, 6),
            "precision": round(float(prec_arr[label_idx]), 6),
            "recall": round(float(rec_arr[label_idx]), 6),
            "f1": round(float(f1_arr[label_idx]), 6),
            "mean_confidence": round(float(np.mean([r["confidence"] for r in subset])), 6),
            "mean_p_true": round(float(np.mean([r["p_true"] for r in subset])), 6),
            "mean_margin": round(float(np.mean([r["margin"] for r in subset])), 6),
            "mean_entropy": round(float(np.mean([r["entropy"] for r in subset])), 6),
            "mean_p_a": round(float(np.mean([r["p_a"] for r in subset])), 6),
            "mean_p_b": round(float(np.mean([r["p_b"] for r in subset])), 6),
        }
    return result


def save_pairwise_outputs(records: list[dict], out_dir: Path) -> None:
    out_path = out_dir / "pairwise_outputs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {out_path} ({len(records)} records)")


def save_summary_metrics(
    all_records: list[dict],
    comparison_id: str,
    version_a: str,
    version_b: str,
    text_field: str,
    out_dir: Path,
) -> None:
    overall = _subset_metrics(all_records, version_b)

    by_split: dict[str, dict] = {}
    for split in ["train", "test"]:
        subset = [r for r in all_records if r["split"] == split]
        if subset:
            by_split[split] = _subset_metrics(subset, version_b)

    by_true_version = _by_version_metrics(all_records, version_a, version_b)

    case_ids = sorted({r["case_id"] for r in all_records if r["case_id"] is not None})
    by_case: list[dict] = []
    for cid in case_ids:
        subset = [r for r in all_records if r["case_id"] == cid]
        m = _subset_metrics(subset, version_b)
        by_case.append({"case_id": cid, **{k: v for k, v in m.items() if k != "case_id"}})

    summary = {
        "comparison_id": comparison_id,
        "version_a": version_a,
        "version_b": version_b,
        "text_field": text_field,
        "overall": overall,
        "by_split": by_split,
        "by_true_version": by_true_version,
        "by_case": by_case,
    }

    out_path = out_dir / "summary_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_path}")


def save_metadata_json(
    args: argparse.Namespace,
    version_a: str,
    version_b: str,
    comparison_id: str,
    label2id: dict[str, int],
    train_examples: list[Example],
    test_examples: list[Example],
    out_dir: Path,
) -> None:
    training_strategy = (
        "qlora" if (args.use_lora and args.load_in_4bit)
        else "lora" if args.use_lora
        else "full_finetune"
    )

    if args.split_mode == "scenario":
        train_cases = sorted({e.scenario for e in train_examples if e.scenario is not None})
        test_cases = sorted({e.scenario for e in test_examples if e.scenario is not None})
        split_spec: dict[str, Any] = {
            "split_mode": "scenario",
            "split_unit": "case",
            "train_cases": train_cases,
            "test_cases": test_cases,
            "cases": sorted(set(train_cases) | set(test_cases)),
            "train_runs": "all",
            "test_runs": "all",
            "split_file": None,
        }
    elif args.split_mode == "batch":
        train_cases = sorted({e.scenario for e in train_examples if e.scenario is not None})
        test_cases = sorted({e.scenario for e in test_examples if e.scenario is not None})
        split_spec = {
            "split_mode": "batch",
            "split_unit": "batch",
            "train_cases": train_cases,
            "test_cases": test_cases,
            "cases": sorted(set(train_cases) | set(test_cases)),
            "train_runs": "all",
            "test_runs": "all",
            "split_file": None,
        }
    else:  # random
        split_spec = {
            "split_mode": "random",
            "split_unit": "sample",
            "train_cases": None,
            "test_cases": None,
            "cases": None,
            "train_runs": "all",
            "test_runs": "all",
            "split_file": None,
        }

    label_map_str = {str(v): k for k, v in label2id.items()}

    metadata: dict[str, Any] = {
        "comparison_id": comparison_id,
        "version_a": version_a,
        "version_b": version_b,
        "task_type": "pairwise_fingerprint_classification",
        "text_field": args.text_field,
        "split_spec": split_spec,
        "model_config": {
            "backbone": args.model_name,
            "training_strategy": training_strategy,
            "use_lora": args.use_lora,
            "load_in_4bit": args.load_in_4bit,
            "max_length": args.max_length,
            "num_labels": 2,
        },
        "training_config": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "label_map": label_map_str,
        "counts": {
            "n_train": len(train_examples),
            "n_test": len(test_examples),
            "n_total": len(train_examples) + len(test_examples),
        },
    }

    out_path = out_dir / "metadata.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA model-fingerprint detector over Agent-Trial diagnosis data.")
    parser.add_argument("--data_dir", type=Path, default=Path("results/generate_diagnosis_distribution"))
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Override output directory. "
                             "Default (multi-class): results/lora_fingerprint/<text_field>/; "
                             "Default (pairwise): results/lora_fingerprint_pairwise/<comparison_id>/")
    parser.add_argument("--text_field", choices=TEXT_FIELD_CHOICES, default="full_dialogue")
    parser.add_argument("--split_mode", choices=["batch", "random", "scenario"], default="batch")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="fraction held out for --split_mode random / scenario")
    # Pairwise version selection
    parser.add_argument("--version_a", default=None,
                        help="First version label for pairwise comparison. "
                             "If omitted and exactly two labels exist, auto-detected.")
    parser.add_argument("--version_b", default=None,
                        help="Second version label for pairwise comparison.")
    parser.add_argument("--comparison_id", default=None,
                        help="Override the comparison_id string used in the output path. "
                             "Default: <version_a>_vs_<version_b>")
    parser.add_argument("--save_text", action="store_true",
                        help="Include full dialogue text in pairwise_outputs.jsonl rows.")
    # Model
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--allow_remote_model_files", action="store_true",
                        help="Allow transformers to download from Hugging Face if not cached locally.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
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
    parser.add_argument("--lora_target_modules", default="auto",
                        help="comma-separated module names LoRA adapts. 'auto' picks per backbone "
                             "(distilbert->q_lin,v_lin ; qwen2/llama->q_proj,v_proj).")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto",
                        help="model dtype. Use bfloat16 for Qwen/Llama LoRA on a GPU.")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="QLoRA: load base in 4-bit (needs CUDA + bitsandbytes). Pairs with --use_lora.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="trade compute for memory; recommended for 7B on a single GPU.")
    parser.add_argument("--prepare_only", action="store_true", help="only export the data split, don't train")
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- load all examples, then optionally filter to two versions ----
    all_examples = load_examples(args.data_dir, text_field=args.text_field)
    if not all_examples:
        raise ValueError(f"No examples with field '{args.text_field}' found under {args.data_dir}")

    all_labels = sorted({e.label_name for e in all_examples})

    # Determine pairwise mode
    version_a: str | None = args.version_a
    version_b: str | None = args.version_b
    pairwise_mode = False

    if version_a is not None and version_b is not None:
        pairwise_mode = True
        for v in (version_a, version_b):
            if v not in all_labels:
                raise ValueError(f"--version_a/--version_b: label '{v}' not found. "
                                 f"Available: {all_labels}")
        all_examples = [e for e in all_examples if e.label_name in {version_a, version_b}]
    elif len(all_labels) == 2:
        pairwise_mode = True
        version_a, version_b = all_labels[0], all_labels[1]
        print(f"[pairwise] auto-detected two labels: version_a={version_a!r}  version_b={version_b!r}")
    elif version_a is not None or version_b is not None:
        raise ValueError("Provide both --version_a and --version_b, or neither.")

    # ---- split ----
    train_examples, test_examples = split_examples(all_examples, args.split_mode, args.test_size, args.seed)
    if not train_examples or not test_examples:
        raise ValueError(
            f"Empty train or test split (mode={args.split_mode}). "
            f"For split_mode=batch each model needs both a _1 and a _2 folder."
        )

    label2id, id2label = build_label_maps(all_examples)
    print_split_summary(train_examples, test_examples)
    print("\nLabels:")
    for label, idx in label2id.items():
        print(f"  {idx}: {label}")

    # ---- output directory ----
    if pairwise_mode:
        comparison_id = args.comparison_id or f"{version_a}_vs_{version_b}"
        output_dir = args.output_dir or (
            Path("results/lora_fingerprint_pairwise") / comparison_id
        )
    else:
        output_dir = args.output_dir or Path("results/lora_fingerprint") / args.text_field

    # ---- prepare-only path ----
    if not pairwise_mode:
        save_dataset_preview(train_examples, test_examples, label2id, output_dir)
    if args.prepare_only:
        if not pairwise_mode:
            print(f"\nPrepared dataset previews in {output_dir}")
        else:
            print(f"\nPairwise comparison: {comparison_id}")
            print(f"  version_a={version_a}  version_b={version_b}")
            print(f"  train={len(train_examples)}  test={len(test_examples)}")
        return

    # ---- build model ----
    local_files_only = not args.allow_remote_model_files
    tokenizer, model = load_backbone(args, label2id, id2label, local_files_only)
    model = maybe_apply_lora(model, args)

    train_ds = TokenizedDataset(train_examples, label2id, tokenizer, args.max_length)
    test_ds = TokenizedDataset(test_examples, label2id, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = choose_device()
    print(f"\nUsing device: {device}")
    metrics, y_true, y_pred = train_model(model, train_loader, test_loader, args, device)

    target_names = [id2label[idx] for idx in sorted(id2label)]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    matrix = confusion_matrix(y_true, y_pred)

    print("\nEvaluation metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(matrix)

    # ---- persist ----
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    if pairwise_mode:
        # Label IDs for version_a and version_b (sorted alphabetically by build_label_maps)
        label_id_a = label2id[version_a]
        label_id_b = label2id[version_b]

        # Non-shuffled loaders for deterministic record collection
        train_loader_eval = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator
        )

        train_records = collect_pairwise_records(
            model, train_examples, train_loader_eval, device,
            label_id_a, label_id_b, "train",
            version_a, version_b, args.data_dir, args.text_field, args.save_text,
        )
        test_records = collect_pairwise_records(
            model, test_examples, test_loader, device,
            label_id_a, label_id_b, "test",
            version_a, version_b, args.data_dir, args.text_field, args.save_text,
        )
        all_records = train_records + test_records

        save_metadata_json(
            args, version_a, version_b, comparison_id,
            label2id, train_examples, test_examples, output_dir,
        )
        save_pairwise_outputs(all_records, output_dir)
        save_summary_metrics(
            all_records, comparison_id, version_a, version_b, args.text_field, output_dir,
        )
        print(f"\nPairwise outputs saved under {output_dir}")
    else:
        # Legacy multi-class outputs
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "text_field": args.text_field,
                    "split_mode": args.split_mode,
                    "model_name": args.model_name,
                    "use_lora": args.use_lora,
                    "n_train": len(train_examples),
                    "n_test": len(test_examples),
                    **metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        header = "," + ",".join(target_names)
        rows = [header] + [
            target_names[i] + "," + ",".join(str(int(v)) for v in matrix[i]) for i in range(len(target_names))
        ]
        (output_dir / "confusion_matrix.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"\nSaved model + metrics under {output_dir}")


if __name__ == "__main__":
    main()
