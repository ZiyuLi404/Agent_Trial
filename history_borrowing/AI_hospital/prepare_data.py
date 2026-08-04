#!/usr/bin/env python3
"""Prepare the first 100 shared AI Hospital cases for history borrowing."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_TO_EVALUATOR = {
    "gpt3": "GPT-3.5-Turbo",
    "gpt4": "GPT-4",
    "qwen_max": "QwenMax",
    "wenxin": "WenXin4",
}
CORRECT_GRADES = {"A", "B"}
GRADE_REWARD = {"A": 1, "B": 1, "C": 0, "D": 0}

CASE_RE = re.compile(r"(\d+)")
DIAGNOSIS_SECTION_RE = re.compile(
    r"#+\s*诊断结果\s*#*\s*[:：]?\s*(.*?)"
    r"(?=\n\s*#+\s*(?:诊断依据|治疗方案|症状|辅助检查|分析总结)\s*#*\s*(?:\n|$)|\Z)",
    re.DOTALL,
)


def case_id(path: Path) -> int:
    match = CASE_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot parse scenario ID from {path}")
    return int(match.group(1))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_references(path: Path) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError("Reference JSON must be the official patient list or a cases list")
    references = {}
    for item in data:
        record = item.get("medical_record") or item.get("raw_medical_record") or {}
        diagnosis = record.get("诊断结果") or record.get("初步诊断")
        if not diagnosis:
            raise ValueError(f"Reference case {item.get('id')} has no diagnosis")
        references[int(item["id"])] = {
            "case_id": int(item["id"]),
            "title": item.get("title", ""),
            "department": item.get("department", ""),
            "reference_diagnosis": str(diagnosis).strip(),
            "reference_source": "AI_Hospital src/data/patients.json",
        }
    return references


def load_evaluations(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    evaluations = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item["doctor_name"]), int(item["patient_id"]))
            evaluations[key] = item
    return evaluations


def extract_diagnosis(text: str) -> tuple[str, str]:
    match = DIAGNOSIS_SECTION_RE.search(text)
    if match and match.group(1).strip():
        return " ".join(match.group(1).split()), "diagnosis_section"
    # Some agents explicitly abstain before producing the requested sections.
    # Preserve that semantic content instead of inventing a diagnosis.
    return " ".join(text.split()), "whole_output_fallback"


def bucket_accuracy(values: list[bool], bucket_size: int) -> dict[str, float]:
    result = {"total": sum(values) / len(values)}
    for start in range(0, len(values), bucket_size):
        bucket = values[start : start + bucket_size]
        if len(bucket) == bucket_size:
            result[f"bucket{start // bucket_size + 1}"] = sum(bucket) / len(bucket)
    return result


def prepare(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    references = load_references(Path(args.reference_json))
    evaluations = load_evaluations(Path(args.evaluation_jsonl))

    models = sorted(MODEL_TO_EVALUATOR)
    paths_by_model: dict[str, dict[int, Path]] = {}
    for model in models:
        model_dir = source_root / model
        paths_by_model[model] = {
            case_id(path): path for path in model_dir.glob("*.json")
        }
    shared_ids = sorted(set.intersection(*(set(v) for v in paths_by_model.values())))
    selected_ids = shared_ids[: args.case_count]
    if len(selected_ids) != args.case_count:
        raise ValueError(
            f"Requested {args.case_count} shared cases but found only {len(selected_ids)}"
        )

    missing_references = [current for current in selected_ids if current not in references]
    if missing_references:
        raise ValueError(f"Missing official references for: {missing_references}")

    reference_subset = {
        "source_url": (
            "https://github.com/LibertFan/AI_Hospital/blob/main/src/data/patients.json"
        ),
        "selection": "first 100 numerically sorted scenario IDs shared by all four models",
        "cases": [references[current] for current in selected_ids],
    }
    write_json(output_root / "official_reference_first_100.json", reference_subset)

    audit_rows: list[dict[str, Any]] = []
    grade_rows: list[dict[str, Any]] = []
    summary_models: dict[str, Any] = {}
    accuracy_by_size = {size: [] for size in args.bucket_sizes}

    for model in models:
        results = []
        outcomes: list[bool] = []
        grades: Counter[str] = Counter()
        extraction_methods: Counter[str] = Counter()
        evaluator_name = MODEL_TO_EVALUATOR[model]

        for current in selected_ids:
            source_path = paths_by_model[model][current]
            source = json.loads(source_path.read_text(encoding="utf-8"))
            samples = source.get("samples")
            if not isinstance(samples, list) or len(samples) != 1:
                raise ValueError(f"Expected one sample in {source_path}")
            sample = samples[0]
            diagnosis_text = str(sample.get("diagnosis_text", "")).strip()
            doctor_dialogue = str(sample.get("doctor_dialogue", "")).strip()
            full_dialogue = str(sample.get("full_dialogue", "")).strip()
            if not diagnosis_text or not doctor_dialogue:
                raise ValueError(f"Missing diagnosis/dialogue in {source_path}")

            extracted, extraction_method = extract_diagnosis(diagnosis_text)
            extraction_methods[extraction_method] += 1
            evaluation = evaluations.get((evaluator_name, current))
            if evaluation is None:
                raise ValueError(f"Missing official evaluation for {model} case {current}")
            official_output = str(
                (evaluation.get("doctor_diagnosis") or {}).get("diagnosis", "")
            ).strip()
            if official_output != diagnosis_text:
                raise ValueError(
                    f"Local output differs from official evaluated output: {model}/{current}"
                )

            grade = evaluation.get("diagnosis_choice")
            grade_source = "official_gpt4_evaluation"
            if grade is None and model == "wenxin" and current == 1201:
                grade = "C"
                grade_source = "imputed_truncated_evaluation"
            if grade not in GRADE_REWARD:
                raise ValueError(f"Invalid/missing diagnosis grade for {model}/{current}")
            correct = bool(GRADE_REWARD[grade])
            grades[grade] += 1
            outcomes.append(correct)

            reference = references[current]["reference_diagnosis"]
            result = {
                "case_id": current,
                "correct_diagnosis": reference,
                "output_diagnosis": extracted,
                "correct": correct,
                "official_diagnosis_grade": grade,
                "grade_source": grade_source,
                "extraction_method": extraction_method,
                "conversation": full_dialogue,
            }
            results.append(result)
            write_json(
                output_root / "embedding_inputs" / model / f"case_{current}.json",
                {
                    "scenario_id": current,
                    "samples": [
                        {
                            "run": int(sample.get("run", 0)),
                            "diagnosis_text": extracted,
                            "doctor_dialogue": doctor_dialogue,
                            "full_dialogue": full_dialogue,
                        }
                    ],
                },
            )
            audit_rows.append(
                {
                    "model": model,
                    "case_id": current,
                    "official_grade": grade,
                    "correct": int(correct),
                    "grade_source": grade_source,
                    "extraction_method": extraction_method,
                    "reference_diagnosis": reference,
                    "output_diagnosis": extracted,
                    "source_file": str(source_path),
                }
            )
            grade_rows.append(
                {
                    "model": model,
                    "evaluator_doctor_name": evaluator_name,
                    "case_id": current,
                    "diagnosis_grade": grade,
                    "binary_correct": int(correct),
                    "grade_source": grade_source,
                }
            )

        accuracy = sum(outcomes) / len(outcomes)
        write_json(
            output_root / "groundtruth" / f"{model}.json",
            {
                "dataset": "AI_Hospital/MVME",
                "source": str(source_root / model),
                "doctor_llm": model,
                "scoring": "official diagnosis grade A/B=correct; C/D=incorrect",
                "total_cases": len(outcomes),
                "correct": sum(outcomes),
                "accuracy": accuracy,
                "cases": selected_ids,
                "results": results,
            },
        )
        summary_models[model] = {
            "total_cases": len(outcomes),
            "correct": sum(outcomes),
            "accuracy": accuracy,
            "official_grade_distribution": dict(sorted(grades.items())),
            "extraction_methods": dict(sorted(extraction_methods.items())),
        }
        for size in args.bucket_sizes:
            accuracy_by_size[size].append(
                {"model": model, **bucket_accuracy(outcomes, size)}
            )

    for size, rows in accuracy_by_size.items():
        bucket_count = args.case_count // size
        write_csv(
            output_root / f"accuracy_by_{size}_cases.csv",
            rows,
            ["model", "total", *[f"bucket{i + 1}" for i in range(bucket_count)]],
        )
    write_csv(
        output_root / "extraction_audit.csv",
        audit_rows,
        [
            "model",
            "case_id",
            "official_grade",
            "correct",
            "grade_source",
            "extraction_method",
            "reference_diagnosis",
            "output_diagnosis",
            "source_file",
        ],
    )
    write_csv(
        output_root / "official_diagnosis_grades_first_100.csv",
        grade_rows,
        [
            "model",
            "evaluator_doctor_name",
            "case_id",
            "diagnosis_grade",
            "binary_correct",
            "grade_source",
        ],
    )
    write_json(output_root / "replicate_map.json", {m: [m] for m in models})
    write_json(
        output_root / "dataset_summary.json",
        {
            "source_root": str(source_root),
            "case_selection": (
                "first 100 numerically sorted scenario IDs in the intersection "
                "of gpt3, gpt4, qwen_max, and wenxin"
            ),
            "available_shared_cases": len(shared_ids),
            "selected_case_ids": selected_ids,
            "models": summary_models,
            "scoring": {
                "source": (
                    "AI_Hospital official GPT-4 five-part evaluation, "
                    "diagnosis_choice"
                ),
                "binary_mapping": {"A": 1, "B": 1, "C": 0, "D": 0},
                "rationale": (
                    "The official rubric describes A as correct and B as "
                    "basically correct; C contains diagnostic errors and D is incorrect."
                ),
                "imputations": [
                    {
                        "model": "wenxin",
                        "case_id": 1201,
                        "grade": "C",
                        "reason": (
                            "The published evaluation was truncated during the "
                            "diagnosis section. The output only partially matched "
                            "the expert diagnoses and was conservatively assigned C."
                        ),
                    }
                ],
            },
            "bucket_sizes": args.bucket_sizes,
        },
    )
    print(
        f"Prepared {len(models)} models x {len(selected_ids)} cases under {output_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_root", default="results/AI_Hospital_fingerprint"
    )
    parser.add_argument(
        "--reference_json", default="/private/tmp/ai_hospital_patients.json"
    )
    parser.add_argument(
        "--evaluation_jsonl",
        default="/private/tmp/evaluation_iiyi_gpt4_5part.jsonl",
    )
    parser.add_argument("--output_root", default="history_borrowing/AI_hospital")
    parser.add_argument("--case_count", type=int, default=100)
    parser.add_argument("--bucket_sizes", type=int, nargs="+", default=[10, 25])
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
