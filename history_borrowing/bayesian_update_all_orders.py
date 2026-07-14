#!/usr/bin/env python3
"""
Simulate Bayesian history borrowing across all possible update orders.

This is a standalone offline analysis script. It does not call LLMs or run new
consultations. It reads either:

  1. an accuracy CSV with model,total,bucket1,... columns, or
  2. user-provided result folders containing case_*.json-style outputs.

For four supplied models it evaluates all 24 possible update sequences:

    model_t0 -> model_t1 -> model_t2 -> model_t3

At each step, previous models contribute similarity-discounted historical prior
information and the current model contributes observed evidence:

    posterior = (m * prior_mean + n * current_mean) / (m + n)
    m_h       = alpha * historical_n_h * exp(-lambda * distance_h)

where distance_h = 1 - similarity_h by default.

Example:
    python history_borrowing/bayesian_update_all_orders.py \
        --accuracy_csv history_borrowing/data/accuracy_by_25_cases.csv \
        --models deepseek-v4-flash deepseek-v4-pro gpt-5_4-mini gpt-5_5 \
        --similarity_modes matrix \
        --similarity_csv history_borrowing/data/similarity_matrix/embedding_diagnosis_similarity_matrix.csv \
        --replicate_map_json history_borrowing/data/results/past_result/replicate_map.json \
        --output_dir history_borrowing/data/results/bayesian_update
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import permutations
from pathlib import Path
from typing import Any


try:
    # Works when run as: python history_borrowing/bayesian_update_all_orders.py
    from history_borrowing import (  # type: ignore
        build_identity_replicate_map,
        collapse_similarity_matrix,
        load_accuracy_csv,
        load_similarity_csv,
    )
except (ImportError, AttributeError):
    # Works when run as: python -m history_borrowing.bayesian_update_all_orders
    from history_borrowing.history_borrowing import (  # type: ignore
        build_identity_replicate_map,
        collapse_similarity_matrix,
        load_accuracy_csv,
        load_similarity_csv,
    )


MODEL_SIMILARITY_MODES = {"embedding", "matrix"}
TEXT_SIMILARITY_MODES = {"patient-only", "diagnosis-only", "full-conversation"}
SIMILARITY_MODES = ["uniform", "matrix", "embedding", *sorted(TEXT_SIMILARITY_MODES)]


@dataclass
class Observation:
    model: str
    unit_id: str
    mean: float
    n: float
    ground_truth: float
    case_id: int | None = None
    bucket_id: str | None = None
    texts: dict[str, str] = field(default_factory=dict)


@dataclass
class RawCase:
    case_id: int
    full_mean: float
    full_n: float
    current_mean: float
    current_n: float
    texts: dict[str, str]


def parse_float_grid(raw: str, name: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as e:
        sys.exit(f"ERROR parsing {name}: {e}")
    if not values:
        sys.exit(f"ERROR: {name} is empty.")
    return values


def parse_case_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def weighted_mean(pairs: list[tuple[float, float]]) -> float:
    total_n = sum(n for _, n in pairs)
    if total_n <= 0:
        return float("nan")
    return sum(v * n for v, n in pairs) / total_n


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split())


def model_from_group(group_name: str) -> str:
    return re.sub(r"_\d+$", "", group_name)


def extract_case_id_from_path(path: Path) -> int | None:
    match = re.search(r"case[_\-]?(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def coerce_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "1", "correct"}:
            return 1.0
        if low in {"false", "no", "0", "incorrect", "wrong"}:
            return 0.0
        try:
            return float(low)
        except ValueError:
            return None
    return None


def find_metric(record: dict[str, Any], metric_field: str) -> float | None:
    aliases = [
        metric_field,
        "correct",
        "is_correct",
        "correctness",
        "accuracy",
        "score",
        "p_correct",
    ]
    seen: set[str] = set()
    for key in aliases:
        if not key or key in seen:
            continue
        seen.add(key)
        if key in record:
            val = coerce_metric(record[key])
            if val is not None:
                return val

    for parent in ("result", "evaluation", "final", "metrics"):
        nested = record.get(parent)
        if isinstance(nested, dict):
            val = find_metric(nested, metric_field)
            if val is not None:
                return val
    return None


def extract_diagnosis_text(record: Any) -> str:
    if isinstance(record, str):
        return clean_text(record)
    if not isinstance(record, dict):
        return ""

    keys = [
        "diagnosis_text",
        "final_diagnosis",
        "diagnosis",
        "output_diagnosis",
        "final_answer",
        "answer",
        "prediction",
        "doctor_answer",
        "doctor_response",
        "response",
        "raw_response",
        "content",
        "text",
        "message",
        "mode_diagnosis",
    ]
    for key in keys:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return clean_text(val)

    for val in record.values():
        if isinstance(val, dict):
            out = extract_diagnosis_text(val)
            if out:
                return out
    return ""


def extract_full_conversation(record: Any) -> str:
    if isinstance(record, str):
        return clean_text(record)
    if not isinstance(record, dict):
        return ""

    keys = [
        "full_dialogue",
        "conversation",
        "consultation",
        "full_conversation",
        "transcript",
        "dialogue",
    ]
    for key in keys:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return clean_text(val)
        if isinstance(val, list):
            parts: list[str] = []
            for item in val:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    role = item.get("role", "")
                    content = item.get("content", item.get("text", item.get("message", "")))
                    if content:
                        parts.append(f"{role}: {content}" if role else str(content))
            text = clean_text(" ".join(parts))
            if text:
                return text
    return ""


def extract_patient_text(full_conversation: str) -> str:
    if not full_conversation:
        return ""
    parts = [
        match.group(1).strip()
        for match in re.finditer(
            r"Patient:\s*(.*?)(?=(?:Doctor:|Measurement:|Patient:|$))",
            full_conversation,
        )
    ]
    if parts:
        return clean_text(" ".join(parts))
    return ""


def extract_text_bundle(record: Any) -> dict[str, str]:
    full = extract_full_conversation(record)
    return {
        "diagnosis-only": extract_diagnosis_text(record),
        "full-conversation": full,
        "patient-only": extract_patient_text(full),
    }


def merge_texts(texts: list[dict[str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for mode in TEXT_SIMILARITY_MODES:
        values = [t.get(mode, "") for t in texts if t.get(mode, "")]
        merged[mode] = clean_text(" ".join(values))
    return merged


def token_similarity(a: str, b: str) -> float:
    a = clean_text(a).lower()
    b = clean_text(b).lower()
    if not a or not b:
        return 0.0
    tokens_a = set(re.findall(r"[a-z0-9]+", a))
    tokens_b = set(re.findall(r"[a-z0-9]+", b))
    if tokens_a and tokens_b:
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        jaccard = 0.0
    if len(a) + len(b) <= 4000:
        seq = SequenceMatcher(None, a, b).ratio()
    else:
        # Full dialogue buckets can contain hundreds of thousands of characters.
        # Token overlap keeps the fallback transparent without quadratic runtime.
        seq = jaccard
    return max(0.0, min(1.0, 0.7 * jaccard + 0.3 * seq))


def safe_float_cell(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARNING: could not read {path}: {e}", file=sys.stderr)
        return None


def iter_scored_json_records(
    obj: Any,
    path: Path,
    metric_field: str,
    max_samples_per_case: int | None,
) -> list[tuple[int, float, float, dict[str, str]]]:
    """Return (case_id, metric_mean, n, texts) records from one JSON object."""
    if obj is None:
        return []

    path_case_id = extract_case_id_from_path(path)

    if isinstance(obj, list):
        out = []
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            cid = item.get("case_id", item.get("scenario_id", path_case_id if path_case_id is not None else i))
            score = find_metric(item, metric_field)
            if score is not None:
                out.append((int(cid), score, 1.0, extract_text_bundle(item)))
        return out

    if not isinstance(obj, dict):
        return []

    for container_key in ("results", "cases", "data", "items", "records"):
        records = obj.get(container_key)
        if isinstance(records, list):
            out = []
            for i, item in enumerate(records):
                if not isinstance(item, dict):
                    continue
                cid = item.get("case_id", item.get("scenario_id", path_case_id if path_case_id is not None else i))
                score = find_metric(item, metric_field)
                if score is not None:
                    out.append((int(cid), score, 1.0, extract_text_bundle(item)))
            return out

    samples = obj.get("samples")
    if isinstance(samples, list):
        cid = int(obj.get("case_id", obj.get("scenario_id", path_case_id if path_case_id is not None else -1)))
        if cid < 0:
            return []
        out = []
        use_samples = samples if max_samples_per_case is None else samples[:max_samples_per_case]
        for sample in use_samples:
            if not isinstance(sample, dict):
                continue
            score = find_metric(sample, metric_field)
            if score is not None:
                out.append((cid, score, 1.0, extract_text_bundle(sample)))
        if out:
            return out

        # Graded distribution summaries may only expose p_correct at top level.
        score = find_metric(obj, metric_field)
        if score is not None:
            n = float(obj.get("runs", obj.get("total_cases", len(samples) or 1)))
            return [(cid, score, n, extract_text_bundle(obj))]
        return []

    cid = obj.get("case_id", obj.get("scenario_id", path_case_id))
    score = find_metric(obj, metric_field)
    if cid is not None and score is not None:
        n = float(obj.get("n", obj.get("runs", obj.get("total_cases", 1))))
        return [(int(cid), score, n, extract_text_bundle(obj))]

    return []


def path_model_name(path: Path, root: Path, models: list[str]) -> str | None:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts

    for part in rel_parts:
        if part in models:
            return part
    for part in rel_parts:
        inferred = model_from_group(part)
        if inferred in models:
            return inferred
    if path.stem in models:
        return path.stem
    inferred_stem = model_from_group(path.stem)
    if inferred_stem in models:
        return inferred_stem
    return None


def parse_model_dirs(items: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"ERROR: --model_dir must be NAME=DIR, got: {item}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name:
            sys.exit(f"ERROR: --model_dir has an empty model name: {item}")
        out[name] = Path(raw_path)
    return out


def load_raw_cases(
    models: list[str],
    data_dirs: list[str] | None,
    model_dir_items: list[str] | None,
    metric_field: str,
    case_ids: set[int] | None,
    case_start: int | None,
    case_end: int | None,
    max_samples_per_case: int | None,
    current_sample_n: int | None,
) -> dict[str, dict[int, RawCase]]:
    model_dirs = parse_model_dirs(model_dir_items)
    unknown_model_dirs = sorted(set(model_dirs) - set(models))
    if unknown_model_dirs:
        sys.exit(f"ERROR: --model_dir supplied names not in --models: {unknown_model_dirs}")

    buckets: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in models}

    candidate_files: list[tuple[str, Path]] = []
    for model, directory in model_dirs.items():
        if directory.is_file():
            candidate_files.append((model, directory))
        elif directory.is_dir():
            for path in sorted(directory.rglob("*.json")):
                if path.name.startswith("._") or path.name == ".DS_Store":
                    continue
                candidate_files.append((model, path))
        else:
            sys.exit(f"ERROR: model_dir path for {model} does not exist: {directory}")

    for raw_root in data_dirs or []:
        root = Path(raw_root)
        if not root.is_dir():
            sys.exit(f"ERROR: data_dir does not exist: {root}")
        for path in sorted(root.rglob("*.json")):
            if path.name.startswith("._") or path.name == ".DS_Store":
                continue
            model = path_model_name(path, root, models)
            if model is not None:
                candidate_files.append((model, path))

    if not candidate_files:
        sys.exit("ERROR: no JSON files found. Supply --data_dir and/or --model_dir.")

    for model, path in candidate_files:
        obj = load_json(path)
        records = iter_scored_json_records(obj, path, metric_field, max_samples_per_case)
        for case_id, score, n, texts in records:
            if case_ids is not None and case_id not in case_ids:
                continue
            if case_ids is None:
                if case_start is not None and case_id < case_start:
                    continue
                if case_end is not None and case_id > case_end:
                    continue
            entry = buckets[model].setdefault(
                case_id,
                {"scores": [], "n": [], "texts": []},
            )
            repeats = max(1, int(round(n)))
            entry["scores"].extend([score] * repeats)
            entry["n"].extend([1.0] * repeats)
            entry["texts"].append(texts)

    out: dict[str, dict[int, RawCase]] = {m: {} for m in models}
    for model in models:
        if not buckets[model]:
            sys.exit(f"ERROR: no scored cases found for model '{model}'.")
        for case_id, entry in buckets[model].items():
            scores = entry["scores"]
            ns = entry["n"]
            if not scores:
                continue
            full_mean = weighted_mean(list(zip(scores, ns)))
            if current_sample_n is not None:
                cur_scores = scores[:current_sample_n]
                cur_ns = ns[:current_sample_n]
            else:
                cur_scores = scores
                cur_ns = ns
            out[model][case_id] = RawCase(
                case_id=case_id,
                full_mean=full_mean,
                full_n=sum(ns),
                current_mean=weighted_mean(list(zip(cur_scores, cur_ns))),
                current_n=sum(cur_ns),
                texts=merge_texts(entry["texts"]),
            )

    return out


def build_fixed_buckets(case_ids: list[int], bucket_size: int, n_buckets: int) -> dict[str, set[int]]:
    buckets: dict[str, set[int]] = {}
    for idx in range(n_buckets):
        start = idx * bucket_size
        end = start + bucket_size
        buckets[f"bucket{idx + 1}"] = set(case_ids[start:end])
    return buckets


def build_equal_buckets(case_ids: list[int], n_buckets: int) -> dict[str, set[int]]:
    buckets: dict[str, set[int]] = {}
    total = len(case_ids)
    for idx in range(n_buckets):
        start = round(idx * total / n_buckets)
        end = round((idx + 1) * total / n_buckets)
        buckets[f"bucket{idx + 1}"] = set(case_ids[start:end])
    return buckets


def parse_bucket_order(raw: str | None, available: list[str]) -> list[str]:
    if raw:
        order = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        order = available[:4]
    if len(order) != 4:
        sys.exit(f"ERROR: bucket order must contain exactly 4 buckets, got: {order}")
    bad = [b for b in order if b not in available]
    if bad:
        sys.exit(f"ERROR: unknown bucket(s) in --bucket_order: {bad}. Available: {available}")
    return order


class AccuracyCsvDataset:
    kind = "accuracy_csv"

    def __init__(
        self,
        accuracy_csv: Path,
        models: list[str],
        bucket_size: int,
        bucket_order_raw: str | None,
    ) -> None:
        all_models, theta_raw, theta_total, bucket_cols = load_accuracy_csv(str(accuracy_csv))
        missing = [m for m in models if m not in all_models]
        if missing:
            sys.exit(f"ERROR: model(s) missing from accuracy CSV: {missing}")
        if len(bucket_cols) < 4:
            sys.exit(f"ERROR: accuracy CSV needs at least 4 bucket columns, found {bucket_cols}")

        self.models = models
        self.bucket_order = parse_bucket_order(bucket_order_raw, bucket_cols)
        self.observed: dict[tuple[str, str], Observation] = {}
        self.ground_truth = {m: theta_total[m] for m in models}

        for model in models:
            for bucket_id in bucket_cols:
                value = theta_raw[model].get(bucket_id)
                if value is None:
                    continue
                self.observed[(model, bucket_id)] = Observation(
                    model=model,
                    unit_id=bucket_id,
                    bucket_id=bucket_id,
                    case_id=None,
                    mean=float(value),
                    n=float(bucket_size),
                    ground_truth=float(theta_total[model]),
                )

    def target_observations(self, order: tuple[str, ...], step_index: int) -> list[Observation]:
        model = order[step_index]
        bucket_id = self.bucket_order[step_index]
        obs = self.observed.get((model, bucket_id))
        if obs is None:
            sys.exit(f"ERROR: no observation for model '{model}', bucket '{bucket_id}'.")
        return [obs]

    def history_observation(
        self,
        history_model: str,
        history_step_index: int,
        target_observation: Observation,
    ) -> Observation | None:
        bucket_id = self.bucket_order[history_step_index]
        return self.observed.get((history_model, bucket_id))


class RawDataset:
    kind = "raw_json"

    def __init__(
        self,
        raw_cases: dict[str, dict[int, RawCase]],
        models: list[str],
        unit: str,
        bucket_size: int,
        bucket_mode: str,
        bucket_order_raw: str | None,
    ) -> None:
        self.raw_cases = raw_cases
        self.models = models
        self.unit = unit
        self.total_by_model = {
            m: weighted_mean([(case.full_mean, case.full_n) for case in raw_cases[m].values()])
            for m in models
        }

        all_case_ids = sorted(set().union(*(set(raw_cases[m]) for m in models)))
        if bucket_mode == "equal":
            self.bucket_cases = build_equal_buckets(all_case_ids, 4)
        else:
            self.bucket_cases = build_fixed_buckets(all_case_ids, bucket_size, 4)
        self.bucket_order = parse_bucket_order(bucket_order_raw, sorted(self.bucket_cases))
        self.bucket_observed = self._build_bucket_observations()

    def _build_bucket_observations(self) -> dict[tuple[str, str], Observation]:
        out: dict[tuple[str, str], Observation] = {}
        for model in self.models:
            model_cases = self.raw_cases[model]
            for bucket_id, case_ids in self.bucket_cases.items():
                cases = [model_cases[cid] for cid in sorted(case_ids) if cid in model_cases]
                if not cases:
                    continue
                out[(model, bucket_id)] = Observation(
                    model=model,
                    unit_id=bucket_id,
                    bucket_id=bucket_id,
                    case_id=None,
                    mean=weighted_mean([(c.current_mean, c.current_n) for c in cases]),
                    n=sum(c.current_n for c in cases),
                    ground_truth=self.total_by_model[model],
                    texts=merge_texts([c.texts for c in cases]),
                )
        return out

    def target_observations(self, order: tuple[str, ...], step_index: int) -> list[Observation]:
        model = order[step_index]
        if self.unit == "bucket":
            bucket_id = self.bucket_order[step_index]
            obs = self.bucket_observed.get((model, bucket_id))
            if obs is None:
                return []
            return [obs]

        observations = []
        for case_id, case in sorted(self.raw_cases[model].items()):
            observations.append(
                Observation(
                    model=model,
                    unit_id=f"case_{case_id}",
                    case_id=case_id,
                    bucket_id=None,
                    mean=case.current_mean,
                    n=case.current_n,
                    ground_truth=case.full_mean,
                    texts=case.texts,
                )
            )
        return observations

    def history_observation(
        self,
        history_model: str,
        history_step_index: int,
        target_observation: Observation,
    ) -> Observation | None:
        if self.unit == "bucket":
            bucket_id = self.bucket_order[history_step_index]
            return self.bucket_observed.get((history_model, bucket_id))

        if target_observation.case_id is None:
            return None
        case = self.raw_cases.get(history_model, {}).get(target_observation.case_id)
        if case is None:
            return None
        return Observation(
            model=history_model,
            unit_id=f"case_{case.case_id}",
            case_id=case.case_id,
            bucket_id=None,
            mean=case.current_mean,
            n=case.current_n,
            ground_truth=case.full_mean,
            texts=case.texts,
        )


def load_replicate_map(raw: str | None) -> dict[str, list[str]] | None:
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(f"ERROR: --replicate_map_json is neither a file nor JSON: {raw}")


def load_model_similarity_matrix(
    similarity_csv: str | None,
    models: list[str],
    replicate_map_json: str | None,
) -> dict[str, dict[str, float]]:
    if not similarity_csv:
        sys.exit("ERROR: --similarity_csv is required for matrix/embedding similarity modes.")

    labels, sim_matrix = load_similarity_csv(similarity_csv)
    replicate_map = load_replicate_map(replicate_map_json)
    if replicate_map is None:
        replicate_map = build_identity_replicate_map(models, labels)
    else:
        missing = [m for m in models if m not in replicate_map]
        if missing:
            sys.exit(f"ERROR: model(s) missing from replicate map: {missing}")
        replicate_map = {m: replicate_map[m] for m in models}

    model_sim, _ = collapse_similarity_matrix(labels, sim_matrix, replicate_map)
    return model_sim


def load_case_similarity_csv(path: str | None) -> dict[int, dict[tuple[str, str], float]]:
    if not path:
        return {}
    out: dict[int, dict[tuple[str, str], float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "case_id" not in reader.fieldnames:
            sys.exit(f"ERROR: case similarity CSV must have a case_id column: {path}")
        pair_cols = [c for c in reader.fieldnames if "_vs_" in c and not c.endswith("_gap")]
        for row in reader:
            try:
                case_id = int(float(row["case_id"]))
            except ValueError:
                continue
            out[case_id] = {}
            for col in pair_cols:
                val = safe_float_cell(row.get(col))
                if val is None:
                    continue
                a, b = col.split("_vs_", 1)
                out[case_id][tuple(sorted((a, b)))] = val
    return out


def lookup_model_similarity(
    matrix: dict[str, dict[str, float]],
    target: str,
    history: str,
) -> float:
    if target in matrix and history in matrix[target]:
        return float(matrix[target][history])
    if history in matrix and target in matrix[history]:
        return float(matrix[history][target])
    sys.exit(f"ERROR: no model similarity value for {history} -> {target}.")


def lookup_case_similarity(
    case_matrix: dict[int, dict[tuple[str, str], float]],
    target: str,
    history: str,
    case_id: int | None,
) -> float | None:
    if case_id is None or case_id not in case_matrix:
        return None
    return case_matrix[case_id].get(tuple(sorted((target, history))))


def similarity_for_pair(
    mode: str,
    target: str,
    history: str,
    target_obs: Observation,
    history_obs: Observation,
    model_matrix: dict[str, dict[str, float]] | None,
    case_matrix: dict[int, dict[tuple[str, str], float]],
) -> float:
    if mode == "uniform":
        return 1.0
    if mode in MODEL_SIMILARITY_MODES:
        if model_matrix is None:
            sys.exit("ERROR: model similarity matrix was not loaded.")
        case_sim = lookup_case_similarity(case_matrix, target, history, target_obs.case_id)
        if case_sim is not None:
            return case_sim
        return lookup_model_similarity(model_matrix, target, history)
    if mode in TEXT_SIMILARITY_MODES:
        return token_similarity(target_obs.texts.get(mode, ""), history_obs.texts.get(mode, ""))
    sys.exit(f"ERROR: unknown similarity mode: {mode}")


def similarity_to_weight(similarity: float, lam: float, similarity_input: str) -> float:
    if similarity_input == "distance":
        distance = max(0.0, similarity)
    else:
        sim = max(0.0, min(1.0, similarity))
        distance = 1.0 - sim
    return math.exp(-lam * distance)


def round_or_blank(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return round(float(value), digits)


def simulate(
    dataset: AccuracyCsvDataset | RawDataset,
    models: list[str],
    similarity_modes: list[str],
    alpha_grid: list[float],
    lambda_grid: list[float],
    similarity_input: str,
    model_matrix: dict[str, dict[str, float]] | None,
    case_matrix: dict[int, dict[tuple[str, str], float]],
    metric_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_orders = list(permutations(models))

    for order_idx, order in enumerate(all_orders, 1):
        order_id = f"order_{order_idx:02d}"
        model_order = " -> ".join(order)

        for step_index, target in enumerate(order):
            target_observations = dataset.target_observations(order, step_index)
            if not target_observations:
                continue

            for mode in similarity_modes:
                for alpha in alpha_grid:
                    for lam in lambda_grid:
                        for target_obs in target_observations:
                            hist_terms: list[tuple[float, float]] = []
                            similarities: list[float] = []
                            raw_history_n = 0.0

                            for hist_step, hist_model in enumerate(order[:step_index]):
                                hist_obs = dataset.history_observation(hist_model, hist_step, target_obs)
                                if hist_obs is None:
                                    continue
                                sim = similarity_for_pair(
                                    mode=mode,
                                    target=target,
                                    history=hist_model,
                                    target_obs=target_obs,
                                    history_obs=hist_obs,
                                    model_matrix=model_matrix,
                                    case_matrix=case_matrix,
                                )
                                weight = similarity_to_weight(sim, lam, similarity_input)
                                effective_n = alpha * hist_obs.n * weight
                                if effective_n <= 0:
                                    continue
                                hist_terms.append((hist_obs.mean, effective_n))
                                similarities.append(sim)
                                raw_history_n += hist_obs.n

                            effective_prior_n = sum(n for _, n in hist_terms)
                            prior_mean = weighted_mean(hist_terms) if hist_terms else float("nan")

                            if effective_prior_n > 0:
                                posterior_mean = (
                                    effective_prior_n * prior_mean
                                    + target_obs.n * target_obs.mean
                                ) / (effective_prior_n + target_obs.n)
                            else:
                                posterior_mean = target_obs.mean

                            signed_error = posterior_mean - target_obs.ground_truth
                            absolute_error = abs(signed_error)

                            rows.append({
                                "order_id": order_id,
                                "model_order": model_order,
                                "step_index": step_index,
                                "history_models": "|".join(order[:step_index]),
                                "target_model": target,
                                "case_id": target_obs.case_id if target_obs.case_id is not None else "",
                                "bucket_id": target_obs.bucket_id or "",
                                "unit_id": target_obs.unit_id,
                                "similarity_mode": mode,
                                "alpha": alpha,
                                "lambda": lam,
                                "prior_mean": round_or_blank(prior_mean),
                                "current_mean": round(target_obs.mean, 6),
                                "posterior_mean": round(posterior_mean, 6),
                                "ground_truth": round(target_obs.ground_truth, 6),
                                "signed_error": round(signed_error, 6),
                                "absolute_error": round(absolute_error, 6),
                                "mae": "",
                                "effective_prior_n": round(effective_prior_n, 6),
                                "current_n": round(target_obs.n, 6),
                                "historical_n": round(raw_history_n, 6),
                                "n_history_models": step_index,
                                "n_history_used": len(hist_terms),
                                "similarity_mean": round_or_blank(mean(similarities) if similarities else float("nan")),
                                "similarity_min": round_or_blank(min(similarities) if similarities else float("nan")),
                                "similarity_max": round_or_blank(max(similarities) if similarities else float("nan")),
                                "metric": metric_name,
                                "input_source": dataset.kind,
                            })

    attach_step_mae(rows)
    return rows


def attach_step_mae(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["order_id"],
            row["step_index"],
            row["target_model"],
            row["similarity_mode"],
            row["alpha"],
            row["lambda"],
        )
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        mae = mean([float(r["absolute_error"]) for r in group_rows])
        for row in group_rows:
            row["mae"] = round(mae, 6)


def summarize_by_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["order_id"], row["model_order"], row["similarity_mode"], row["alpha"], row["lambda"])
        groups.setdefault(key, []).append(row)

    out = []
    for (order_id, model_order, mode, alpha, lam), group_rows in sorted(groups.items()):
        final_step = max(int(r["step_index"]) for r in group_rows)
        final_rows = [r for r in group_rows if int(r["step_index"]) == final_step]
        out.append({
            "order_id": order_id,
            "model_order": model_order,
            "similarity_mode": mode,
            "alpha": alpha,
            "lambda": lam,
            "mae": round(mean([float(r["absolute_error"]) for r in group_rows]), 6),
            "final_step_mae": round(mean([float(r["absolute_error"]) for r in final_rows]), 6),
            "mean_effective_prior_n": round(mean([float(r["effective_prior_n"]) for r in group_rows]), 6),
            "mean_current_n": round(mean([float(r["current_n"]) for r in group_rows]), 6),
            "n_rows": len(group_rows),
            "n_steps": len({r["step_index"] for r in group_rows}),
        })
    return out


def summarize_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["target_model"], row["similarity_mode"], row["alpha"], row["lambda"])
        groups.setdefault(key, []).append(row)

    out = []
    for (target_model, mode, alpha, lam), group_rows in sorted(groups.items()):
        out.append({
            "target_model": target_model,
            "similarity_mode": mode,
            "alpha": alpha,
            "lambda": lam,
            "mae": round(mean([float(r["absolute_error"]) for r in group_rows]), 6),
            "mean_signed_error": round(mean([float(r["signed_error"]) for r in group_rows]), 6),
            "mean_effective_prior_n": round(mean([float(r["effective_prior_n"]) for r in group_rows]), 6),
            "mean_current_n": round(mean([float(r["current_n"]) for r in group_rows]), 6),
            "mean_step_index": round(mean([float(r["step_index"]) for r in group_rows]), 6),
            "n_rows": len(group_rows),
            "n_orders": len({r["order_id"] for r in group_rows}),
        })
    return out


def best_orders(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (row["similarity_mode"], row["alpha"], row["lambda"])
        groups.setdefault(key, []).append(row)

    out = []
    for key, group_rows in sorted(groups.items()):
        ranked = sorted(group_rows, key=lambda r: (float(r["mae"]), str(r["order_id"])))
        for rank, row in enumerate(ranked, 1):
            new_row = dict(row)
            new_row["rank"] = rank
            out.append(new_row)
    return out


def best_global_setting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["similarity_mode"], row["alpha"], row["lambda"])
        groups.setdefault(key, []).append(row)

    best_key = min(
        groups,
        key=lambda key: mean([float(r["absolute_error"]) for r in groups[key]]),
    )
    return {
        "similarity_mode": best_key[0],
        "alpha": best_key[1],
        "lambda": best_key[2],
        "mae": mean([float(r["absolute_error"]) for r in groups[best_key]]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_models(models: list[str]) -> None:
    if len(models) != 4:
        sys.exit(f"ERROR: exactly 4 models are required, got {len(models)}: {models}")
    if len(set(models)) != 4:
        sys.exit(f"ERROR: model names must be unique: {models}")


def validate_similarity_modes(modes: list[str]) -> None:
    bad = [m for m in modes if m not in SIMILARITY_MODES]
    if bad:
        sys.exit(f"ERROR: unknown similarity mode(s): {bad}. Choices: {SIMILARITY_MODES}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian updating/history borrowing across all 24 orders of 4 models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--accuracy_csv",
        type=str,
        help="CSV with columns model,total,bucket1,... from history_borrowing/accuracy_summary.py.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        help="One or more roots containing model/group case JSON files.",
    )

    parser.add_argument("--model_dir", action="append", default=None, metavar="MODEL=PATH")
    parser.add_argument("--models", nargs=4, required=True, help="Exactly four model names.")
    parser.add_argument(
        "--similarity_modes",
        nargs="+",
        default=["uniform"],
        help=f"One or more modes. Choices: {', '.join(SIMILARITY_MODES)}",
    )
    parser.add_argument("--similarity_csv", default=None, help="Model similarity/distance matrix CSV.")
    parser.add_argument("--case_similarity_csv", default=None, help="Optional case-level pairwise similarity CSV.")
    parser.add_argument("--replicate_map_json", default=None, help="JSON file/string mapping model names to matrix labels.")
    parser.add_argument(
        "--similarity_input",
        choices=["similarity", "distance"],
        default="similarity",
        help="Interpret --similarity_csv values as similarities or distances.",
    )

    parser.add_argument("--alpha_grid", default="1.0", help="Comma-separated alpha values.")
    parser.add_argument("--lambda_grid", default="0.0", help="Comma-separated lambda values.")
    parser.add_argument("--bucket_order", default=None, help="Comma-separated bucket names assigned to steps 0..3.")
    parser.add_argument("--bucket_size", type=int, default=25, help="Sample count for each accuracy CSV/fixed bucket.")
    parser.add_argument("--bucket_mode", choices=["fixed", "equal"], default="fixed")
    parser.add_argument("--unit", choices=["bucket", "case"], default="bucket", help="Raw JSON evaluation unit.")

    parser.add_argument("--metric_field", default="correctness", help="Metric/correctness field to read from JSON.")
    parser.add_argument("--case_ids", default=None, help="Comma/range case filter, e.g. 2,5,15-18.")
    parser.add_argument("--case_start", type=int, default=None)
    parser.add_argument("--case_end", type=int, default=None)
    parser.add_argument("--max_samples_per_case", type=int, default=None)
    parser.add_argument(
        "--current_sample_n",
        type=int,
        default=None,
        help="For repeated-sample raw JSON, use only first N samples as current evidence.",
    )

    parser.add_argument("--output_dir", default="bayesian_update_outputs")
    parser.add_argument("--no_best_orders", action="store_true")
    args = parser.parse_args()

    models = list(args.models)
    validate_models(models)
    validate_similarity_modes(args.similarity_modes)

    alpha_grid = parse_float_grid(args.alpha_grid, "--alpha_grid")
    lambda_grid = parse_float_grid(args.lambda_grid, "--lambda_grid")

    if args.accuracy_csv and (args.data_dir or args.model_dir):
        sys.exit("ERROR: use either --accuracy_csv or raw JSON inputs (--data_dir/--model_dir), not both.")
    if not args.accuracy_csv and not args.data_dir and not args.model_dir:
        sys.exit("ERROR: provide --accuracy_csv or raw JSON inputs via --data_dir/--model_dir.")

    if args.accuracy_csv:
        dataset: AccuracyCsvDataset | RawDataset = AccuracyCsvDataset(
            accuracy_csv=Path(args.accuracy_csv),
            models=models,
            bucket_size=args.bucket_size,
            bucket_order_raw=args.bucket_order,
        )
        if any(mode in TEXT_SIMILARITY_MODES for mode in args.similarity_modes):
            sys.exit("ERROR: text similarity modes require raw JSON input via --data_dir/--model_dir.")
    else:
        raw_cases = load_raw_cases(
            models=models,
            data_dirs=args.data_dir,
            model_dir_items=args.model_dir,
            metric_field=args.metric_field,
            case_ids=parse_case_ids(args.case_ids),
            case_start=args.case_start,
            case_end=args.case_end,
            max_samples_per_case=args.max_samples_per_case,
            current_sample_n=args.current_sample_n,
        )
        dataset = RawDataset(
            raw_cases=raw_cases,
            models=models,
            unit=args.unit,
            bucket_size=args.bucket_size,
            bucket_mode=args.bucket_mode,
            bucket_order_raw=args.bucket_order,
        )

    need_model_matrix = any(mode in MODEL_SIMILARITY_MODES for mode in args.similarity_modes)
    model_matrix = (
        load_model_similarity_matrix(args.similarity_csv, models, args.replicate_map_json)
        if need_model_matrix
        else None
    )
    case_matrix = load_case_similarity_csv(args.case_similarity_csv)

    rows = simulate(
        dataset=dataset,
        models=models,
        similarity_modes=args.similarity_modes,
        alpha_grid=alpha_grid,
        lambda_grid=lambda_grid,
        similarity_input=args.similarity_input,
        model_matrix=model_matrix,
        case_matrix=case_matrix,
        metric_name=args.metric_field,
    )
    if not rows:
        sys.exit("ERROR: simulation produced no rows.")

    summary_order = summarize_by_order(rows)
    summary_model = summarize_by_model(rows)
    best = best_orders(summary_order)
    best_global = best_global_setting(rows)
    best_order_mae = min(float(r["mae"]) for r in summary_order)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "bayesian_update_all_orders.csv", rows)
    write_csv(output_dir / "bayesian_update_summary_by_order.csv", summary_order)
    write_csv(output_dir / "bayesian_update_summary_by_model.csv", summary_model)
    if not args.no_best_orders:
        write_csv(output_dir / "best_orders_by_mae.csv", best)

    print(f"Processed {len(models)} models, 24 orders, {len(rows)} detail row(s).")
    print(f"Wrote outputs to: {output_dir}")
    print(
        "Best global hyperparameter MAE: "
        f"{best_global['mae']:.6f} "
        f"(mode={best_global['similarity_mode']}, "
        f"alpha={best_global['alpha']}, lambda={best_global['lambda']})"
    )
    print(f"Best single-order MAE: {best_order_mae:.6f}")


if __name__ == "__main__":
    main()
