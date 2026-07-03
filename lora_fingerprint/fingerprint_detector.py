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

依赖：torch / transformers / scikit-learn / numpy（仓库已装）。
      --use_lora 时额外需要 `peft`（pip install peft），不开则完全不需要。

示例：
    python lora_fingerprint/fingerprint_detector.py \
        --text_field full_dialogue --epochs 5 --batch_size 4
    python lora_fingerprint/fingerprint_detector.py \
        --text_field diagnosis_text --use_lora --lora_r 16
    python lora_fingerprint/fingerprint_detector.py --prepare_only   # 只看数据划分，不训练
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
    # Optional extras (not used by the classifier itself) so downstream
    # consumers such as lora_fingerprint/pairwise_fingerprint.py can report
    # per-sample diagnosis text without re-parsing the case JSON.
    diagnosis_text: str | None = None
    gold_diagnosis: str | None = None


def scenario_of(case_file: str) -> int | None:
    """从 case_<n>.json 解析病例(scenario)编号。"""
    m = re.search(r"case_(\d+)", case_file)
    return int(m.group(1)) if m else None


def family_from_dir(dirname: str) -> str:
    """模型家族 = 文件夹名去掉尾部批次号 _<digits>。无尾号时原样返回。"""
    return BATCH_SUFFIX_RE.sub("", dirname)


def batch_from_dir(dirname: str) -> str | None:
    """尾部批次号（'1' / '2' ...），用于 split_mode=batch。"""
    m = re.search(r"_(\d+)$", dirname)
    return m.group(1) if m else None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return "\n".join(str(value).replace("\r", "\n").splitlines()).strip()


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
            gold_diagnosis = normalize_text(case_obj.get("correct_diagnosis_reference")) or None
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                text = normalize_text(sample.get(text_field))
                if not text:
                    continue
                examples.append(
                    Example(
                        text=text,
                        label_name=label_name,
                        source_dir=model_dir.name,
                        case_file=case_path.name,
                        run=sample.get("run"),
                        scenario=scenario_of(case_path.name),
                        diagnosis_text=normalize_text(sample.get("diagnosis_text")) or None,
                        gold_diagnosis=gold_diagnosis,
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
        # 病例完全留出：训练与测试用互不重叠的 scenario（跨批次合并）。
        # 这才测「在没见过的病例上还能不能认出版本」。
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

    # random, stratified by label
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
# Torch dataset (no HuggingFace `datasets` dependency)
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
# LoRA target modules per backbone family. "auto" picks from model_type;
# distilbert isn't in peft's built-in mapping so it must be named explicitly.
LORA_TARGETS_BY_TYPE = {
    "distilbert": ["q_lin", "v_lin"],
    "bert": ["query", "value"],
    "qwen2": ["q_proj", "v_proj"],          # Qwen2 / Qwen2.5
    "qwen3": ["q_proj", "v_proj"],
    "llama": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
}


def resolve_lora_targets(model: Any, spec: str) -> list[str] | None:
    if spec and spec != "auto":
        return [m.strip() for m in spec.split(",") if m.strip()]
    model_type = getattr(model.config, "model_type", "")
    return LORA_TARGETS_BY_TYPE.get(model_type)  # None -> let peft auto-resolve


def maybe_apply_lora(model: Any, args: argparse.Namespace) -> Any:
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "You passed --use_lora but `peft` is not installed. "
            "Run `pip install peft`, or drop --use_lora to fine-tune the full model."
        ) from exc

    # QLoRA: stabilise a 4-bit base before attaching adapters.
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
    """Load tokenizer + sequence-classification model.

    Works for encoders (DistilBERT/BERT) and decoder-only LMs (Qwen2.5/Llama):
    a decoder LM needs a pad token and config.pad_token_id so the classifier
    can locate the last real token. Supports fp16/bf16 and 4-bit (QLoRA).
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:  # most causal LMs (Qwen/Llama) ship without one
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
    """True if accelerate already placed the model (device_map / 4-bit)."""
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

    # A 4-bit / device_map model is already placed by accelerate — don't move it.
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
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA model-fingerprint detector over Agent-Trial diagnosis data.")
    parser.add_argument("--data_dir", type=Path, default=Path("results/generate_diagnosis_distribution"))
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="default: results/lora_fingerprint/<text_field>/")
    parser.add_argument("--text_field", choices=TEXT_FIELD_CHOICES, default="full_dialogue")
    parser.add_argument("--split_mode", choices=["batch", "random", "scenario"], default="batch")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="fraction held out for --split_mode random / scenario")
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

    output_dir = args.output_dir or Path("results/lora_fingerprint") / args.text_field

    set_seed(args.seed)

    examples = load_examples(args.data_dir, text_field=args.text_field)
    if not examples:
        raise ValueError(f"No examples with field '{args.text_field}' found under {args.data_dir}")

    train_examples, test_examples = split_examples(examples, args.split_mode, args.test_size, args.seed)
    if not train_examples or not test_examples:
        raise ValueError(
            f"Empty train or test split (mode={args.split_mode}). "
            f"For split_mode=batch each model needs both a _1 and a _2 folder."
        )

    label2id, id2label = build_label_maps(examples)
    print_split_summary(train_examples, test_examples)
    print("\nLabels:")
    for label, idx in label2id.items():
        print(f"  {idx}: {label}")

    save_dataset_preview(train_examples, test_examples, label2id, output_dir)
    if args.prepare_only:
        print(f"\nPrepared dataset previews in {output_dir}")
        return

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

    # ---- persist (results/**/*.json|model weights are git-ignored & regenerable) ----
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
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
    