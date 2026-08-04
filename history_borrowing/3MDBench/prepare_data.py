#!/usr/bin/env python3
"""Prepare the 80-case assessment-only 3MDBench export for history borrowing."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_TO_MODEL = {
    "deepseek_flash_doctor": "deepseek_flash",
    "deepseek_pro_doctor": "deepseek_pro",
    "gpt5_mini_doctor": "gpt5_mini",
    "qwen_plus_doctor": "qwen_plus",
}
RUBRIC_FIELDS = ("1.1", "1.2", "1.3", "2.1", "2.2", "3.1", "3.2")
PASSING_COMPETENCE = {"satisfactory", "excellent"}
CASE_RE = re.compile(r"case_(\d+)\.json$")


def case_id(path: Path) -> int:
    match = CASE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse case ID from {path}")
    return int(match.group(1))


def extract_assessment(raw: str) -> dict[str, Any]:
    values: dict[str, int | None] = {}
    for field in RUBRIC_FIELDS:
        match = re.search(
            rf"[\"']{re.escape(field)}[\"']\s*:\s*([01])", raw
        )
        values[field] = int(match.group(1)) if match else None
    competence_match = re.search(
        r"[\"']4\.1[\"']\s*:\s*[\"']"
        r"(unsatisfactory|satisfactory|excellent)[\"']",
        raw,
        re.IGNORECASE,
    )
    competence = competence_match.group(1).lower() if competence_match else None
    observed = sum(value is not None for value in values.values()) + int(
        competence is not None
    )
    status = "complete" if observed == 8 else "partial" if observed else "missing"
    parts = [f"assessment_status={status}"]
    parts.extend(
        f"rubric_{field}={values[field] if values[field] is not None else 'missing'}"
        for field in RUBRIC_FIELDS
    )
    parts.append(f"overall_competence={competence or 'missing'}")
    return {
        "rubric_values": values,
        "overall_competence": competence,
        "assessment_status": status,
        "assessment_text": "; ".join(parts),
        "correct": competence in PASSING_COMPETENCE,
    }


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


def clear_stale_generated_outputs(output_root: Path) -> None:
    for name in ("groundtruth", "embedding_inputs", "similarity_matrix", "results"):
        path = output_root / name
        if path.exists():
            shutil.rmtree(path)
    for name in (
        "accuracy_by_10_cases.csv",
        "accuracy_by_20_cases.csv",
        "accuracy_by_25_cases.csv",
        "accuracy_by_version_bundle.csv",
        "extraction_audit.csv",
        "assessment_audit.csv",
        "dataset_summary.json",
        "replicate_map.json",
    ):
        path = output_root / name
        if path.exists():
            path.unlink()


def prepare(source_root: Path, output_root: Path) -> None:
    paths_by_model: dict[str, dict[int, Path]] = {}
    for source_name, model in SOURCE_TO_MODEL.items():
        source_dir = source_root / source_name
        paths = {case_id(path): path for path in source_dir.glob("case_*.json")}
        if not paths:
            raise ValueError(f"No case files found under {source_dir}")
        paths_by_model[model] = paths
    shared_ids = sorted(set.intersection(*(set(paths) for paths in paths_by_model.values())))
    if not shared_ids:
        raise ValueError("The four model directories have no shared cases")
    if any(set(paths) != set(shared_ids) for paths in paths_by_model.values()):
        raise ValueError("Every model must have exactly the same case IDs")

    # These are generated artifacts from the previous source export. Clear them
    # only after the replacement input has passed coverage validation.
    clear_stale_generated_outputs(output_root)

    audit_rows: list[dict[str, Any]] = []
    summary_models: dict[str, Any] = {}
    for model, paths in paths_by_model.items():
        results = []
        competence_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        correct_count = 0
        for current in shared_ids:
            path = paths[current]
            source = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(source, dict) or len(source) != 1:
                raise ValueError(f"Expected one top-level case in {path}")
            key, record = next(iter(source.items()))
            if str(key) != str(current) or not isinstance(record, dict):
                raise ValueError(f"Case ID/schema mismatch in {path}")
            raw = str(record.get("assessment", ""))
            parsed = extract_assessment(raw)
            competence = parsed["overall_competence"] or "missing"
            correct = bool(parsed["correct"])
            correct_count += int(correct)
            competence_counts[competence] += 1
            status_counts[str(parsed["assessment_status"])] += 1
            result = {
                "case_id": current,
                "correct": correct,
                "score": int(correct),
                "overall_clinical_competence": parsed["overall_competence"],
                "assessment_status": parsed["assessment_status"],
                "rubric_values": parsed["rubric_values"],
                "assessment_text": parsed["assessment_text"],
                "raw_assessment": raw,
                "source_file": str(path),
            }
            results.append(result)
            write_json(
                output_root / "embedding_inputs" / model / f"case_{current}.json",
                {
                    "scenario_id": current,
                    "samples": [
                        {
                            "run": 0,
                            "assessment_text": parsed["assessment_text"],
                        }
                    ],
                },
            )
            audit_rows.append(
                {
                    "model": model,
                    "case_id": current,
                    "assessment_status": parsed["assessment_status"],
                    "overall_competence": competence,
                    "correct": int(correct),
                    **{
                        f"rubric_{field}": (
                            parsed["rubric_values"][field]
                            if parsed["rubric_values"][field] is not None
                            else ""
                        )
                        for field in RUBRIC_FIELDS
                    },
                    "source_file": str(path),
                }
            )
        write_json(
            output_root / "groundtruth" / f"{model}.json",
            {
                "dataset": "3MDBench assessment export",
                "source": str(source_root),
                "doctor_llm": model,
                "scoring": (
                    "overall clinical competence satisfactory/excellent=1; "
                    "unsatisfactory/missing=0"
                ),
                "total_cases": len(results),
                "correct": correct_count,
                "accuracy": correct_count / len(results),
                "cases": shared_ids,
                "results": results,
            },
        )
        summary_models[model] = {
            "total_cases": len(results),
            "correct": correct_count,
            "accuracy": correct_count / len(results),
            "competence_distribution": dict(sorted(competence_counts.items())),
            "assessment_status": dict(sorted(status_counts.items())),
        }

    write_csv(
        output_root / "assessment_audit.csv",
        audit_rows,
        [
            "model",
            "case_id",
            "assessment_status",
            "overall_competence",
            "correct",
            *[f"rubric_{field}" for field in RUBRIC_FIELDS],
            "source_file",
        ],
    )
    write_json(
        output_root / "replicate_map.json", {model: [model] for model in paths_by_model}
    )
    write_json(
        output_root / "dataset_summary.json",
        {
            "source_root": str(source_root),
            "source_schema": "assessment-only",
            "models": summary_models,
            "shared_case_ids": shared_ids,
            "n_shared_cases": len(shared_ids),
            "outcome_definition": (
                "Overall Clinical Competence 4.1: satisfactory/excellent=1; "
                "unsatisfactory or missing/unparseable=0"
            ),
            "embedding_definition": (
                "Canonical assessment status, seven binary rubric items, and "
                "overall competence; missing fields remain explicit."
            ),
            "warnings": [
                "The updated source contains assessments, not doctor dialogues or diagnoses.",
                "Missing or truncated assessments are conservatively scored as failures.",
                "Assessment embedding similarity measures evaluator-profile similarity, not dialogue semantic similarity.",
            ],
        },
    )
    print(
        f"Prepared {len(paths_by_model)} models x {len(shared_ids)} assessment cases "
        f"under {output_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="results/3MDBench")
    parser.add_argument("--output_root", default="history_borrowing/3MDBench")
    args = parser.parse_args()
    prepare(Path(args.source_root), Path(args.output_root))


if __name__ == "__main__":
    main()
