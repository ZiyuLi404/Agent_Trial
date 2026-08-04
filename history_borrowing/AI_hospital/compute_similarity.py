#!/usr/bin/env python3
"""Compute same-case Qwen embedding similarity for prepared AI Hospital data."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


TEXT_FIELDS = ("diagnosis_text", "doctor_dialogue", "full_dialogue")


def case_id(path: Path) -> int:
    match = re.search(r"case_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Unrecognized case filename: {path}")
    return int(match.group(1))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(input_dir: Path, field: str) -> list[dict[str, object]]:
    rows = []
    for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for path in sorted(model_dir.glob("case_*.json"), key=case_id):
            data = json.loads(path.read_text(encoding="utf-8"))
            samples = data.get("samples")
            if not isinstance(samples, list) or len(samples) != 1:
                raise ValueError(f"Expected exactly one sample in {path}")
            text = samples[0].get(field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing {field} in {path}")
            rows.append(
                {
                    "model": model_dir.name,
                    "case_id": case_id(path),
                    "text": " ".join(text.split()),
                    "source_file": str(path),
                }
            )
    models = sorted({str(row["model"]) for row in rows})
    case_sets = {
        model: {int(row["case_id"]) for row in rows if row["model"] == model}
        for model in models
    }
    if not rows or any(case_sets[m] != case_sets[models[0]] for m in models[1:]):
        raise ValueError("Every model must contain the same non-empty case set")
    return rows


def compute(args: argparse.Namespace) -> None:
    rows = load_rows(Path(args.input_dir), args.text_field)
    texts = [str(row["text"]) for row in rows]
    encoder = SentenceTransformer(args.model)
    encoder.max_seq_length = args.max_seq_length
    embeddings = np.asarray(
        encoder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=args.batch_size,
            show_progress_bar=True,
        )
    )
    if embeddings.shape[0] != len(rows):
        raise ValueError("Embedding count does not match row count")

    output_dir = Path(args.output_dir)
    models = sorted({str(row["model"]) for row in rows})
    cases = sorted({int(row["case_id"]) for row in rows})
    pairs = list(itertools.combinations(models, 2))
    vectors = {
        (str(row["model"]), int(row["case_id"])): embeddings[index]
        for index, row in enumerate(rows)
    }
    values = {pair: [] for pair in pairs}
    case_rows = []
    for current in cases:
        row: dict[str, object] = {"case_id": current}
        for left, right in pairs:
            similarity = float(
                np.clip(
                    np.dot(vectors[(left, current)], vectors[(right, current)]),
                    -1.0,
                    1.0,
                )
            )
            row[f"{left}_vs_{right}"] = similarity
            values[(left, right)].append(similarity)
        case_rows.append(row)
    pair_fields = [f"{left}_vs_{right}" for left, right in pairs]
    write_csv(
        output_dir / "case_level_model_similarities.csv",
        case_rows,
        ["case_id", *pair_fields],
    )

    summary_rows = []
    for (left, right), pair_values in values.items():
        summary_rows.append(
            {
                "comparison": f"{left}_vs_{right}",
                "mean_case_level_similarity": float(np.mean(pair_values)),
                "std": float(np.std(pair_values, ddof=1)),
                "min": min(pair_values),
                "max": max(pair_values),
                "n_cases": len(pair_values),
                "mean_embedding_gap": 1.0 - float(np.mean(pair_values)),
            }
        )
    write_csv(
        output_dir / "summary_model_similarities.csv",
        summary_rows,
        [
            "comparison",
            "mean_case_level_similarity",
            "std",
            "min",
            "max",
            "n_cases",
            "mean_embedding_gap",
        ],
    )

    matrix_rows = []
    for left in models:
        row: dict[str, object] = {"model": left}
        for right in models:
            if left == right:
                row[right] = 1.0
            else:
                pair = (left, right) if (left, right) in values else (right, left)
                row[right] = float(np.mean(values[pair]))
        matrix_rows.append(row)
    write_csv(
        output_dir / "mean_model_similarity_matrix.csv",
        matrix_rows,
        ["model", *models],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "case_model_embeddings.npz",
        **{
            f"case_{row['case_id']}__{row['model']}": embeddings[index]
            for index, row in enumerate(rows)
        },
    )
    write_csv(
        output_dir / "raw_outputs.csv",
        rows,
        ["model", "case_id", "text", "source_file"],
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "embedding_model": args.model,
                "text_field": args.text_field,
                "max_seq_length": args.max_seq_length,
                "batch_size": args.batch_size,
                "models": models,
                "case_ids": cases,
                "n_texts": len(rows),
                "embedding_dimension": int(embeddings.shape[1]),
                "aggregation": "mean of same-case pairwise cosine similarities",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.text_field} similarity to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", default="history_borrowing/AI_hospital/embedding_inputs"
    )
    parser.add_argument("--text_field", required=True, choices=TEXT_FIELDS)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    compute(parse_args())
