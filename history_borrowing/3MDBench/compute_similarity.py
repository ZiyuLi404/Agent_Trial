#!/usr/bin/env python3
"""Compute batched, same-case embedding similarity for prepared 3MDBench data."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


TEXT_FIELDS = ("diagnosis_text", "full_dialogue", "assessment_text")


def case_id(path: Path) -> int:
    match = re.search(r"case_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Unrecognized case filename: {path}")
    return int(match.group(1))


def load_rows(input_dir: Path, text_field: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for path in sorted(model_dir.glob("case_*.json"), key=case_id):
            data = json.loads(path.read_text(encoding="utf-8"))
            samples = data.get("samples")
            if not isinstance(samples, list) or len(samples) != 1:
                raise ValueError(f"Expected exactly one prepared sample in {path}")
            text = samples[0].get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing '{text_field}' in {path}")
            rows.append(
                {
                    "model": model_dir.name,
                    "case_id": case_id(path),
                    "text": " ".join(text.split()),
                    "source_file": str(path),
                }
            )
    if not rows:
        raise ValueError(f"No prepared cases found under {input_dir}")

    models = sorted({str(row["model"]) for row in rows})
    case_sets = {
        model: {int(row["case_id"]) for row in rows if row["model"] == model}
        for model in models
    }
    if any(case_sets[model] != case_sets[models[0]] for model in models[1:]):
        raise ValueError("Every model must contain the same case IDs")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rows = load_rows(input_dir, args.text_field)
    texts = [str(row["text"]) for row in rows]

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings)
    if embeddings.shape[0] != len(rows):
        raise ValueError("Embedding count does not match prepared row count")

    vector_by_key = {
        (str(row["model"]), int(row["case_id"])): embeddings[index]
        for index, row in enumerate(rows)
    }
    models = sorted({str(row["model"]) for row in rows})
    cases = sorted({int(row["case_id"]) for row in rows})
    pairs = list(itertools.combinations(models, 2))

    case_rows: list[dict[str, object]] = []
    pair_values: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
    for current_case in cases:
        case_row: dict[str, object] = {"case_id": current_case}
        for left, right in pairs:
            similarity = float(
                np.clip(
                    np.dot(
                        vector_by_key[(left, current_case)],
                        vector_by_key[(right, current_case)],
                    ),
                    -1.0,
                    1.0,
                )
            )
            case_row[f"{left}_vs_{right}"] = similarity
            pair_values[(left, right)].append(similarity)
        case_rows.append(case_row)

    pair_columns = [f"{left}_vs_{right}" for left, right in pairs]
    write_csv(
        output_dir / "case_level_model_similarities.csv",
        case_rows,
        ["case_id", *pair_columns],
    )

    summary_rows: list[dict[str, object]] = []
    for pair, values in pair_values.items():
        summary_rows.append(
            {
                "comparison": f"{pair[0]}_vs_{pair[1]}",
                "mean_case_level_similarity": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n_cases": len(values),
                "mean_embedding_gap": 1.0 - float(np.mean(values)),
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

    matrix_rows: list[dict[str, object]] = []
    for left in models:
        matrix_row: dict[str, object] = {"model": left}
        for right in models:
            if left == right:
                matrix_row[right] = 1.0
            else:
                pair = (left, right) if (left, right) in pair_values else (right, left)
                matrix_row[right] = float(np.mean(pair_values[pair]))
        matrix_rows.append(matrix_row)
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
    metadata = {
        "embedding_model": args.model,
        "text_field": args.text_field,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "models": models,
        "case_ids": cases,
        "n_texts": len(rows),
        "embedding_dimension": int(embeddings.shape[1]),
        "aggregation": "mean of same-case pairwise cosine similarities",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.text_field} similarity for {len(models)} models to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        default="history_borrowing/3MDBench/embedding_inputs",
    )
    parser.add_argument("--text_field", required=True, choices=TEXT_FIELDS)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_seq_length <= 0:
        raise SystemExit("ERROR: batch_size and max_seq_length must be positive")
    compute(args)


if __name__ == "__main__":
    main()
