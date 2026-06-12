#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case-level embedding analysis for Data2.zip.

This script follows the same case-level embedding method used before:

For each anchor case and each group folder:
  1. Read diagnosis outputs from case_x.json.
  2. Use the first N outputs per group/case file, default N=10.
  3. Embed each diagnosis output.
  4. Average embeddings within the same case and group.
  5. Normalize the averaged embedding.

Then:
  A. Group-level comparison:
     compare every group pair, e.g.
       deepseek_pro_1 vs deepseek_pro_2
       Qwen_plus_turbo_1 vs Qwen_plus_turbo_2
       deepseek_flash_1 vs deepseek_pro_1
       deepseek_flash_1 vs Qwen_plus_turbo_1
       ...

  B. Model-level comparison:
     merge groups by model type:
       folders containing "pro"   -> pro
       folders containing "flash" or "falsh" -> flash
       folders containing "qwen"  -> qwen
     For each case and each model, average all outputs from that model,
     then compare:
       flash vs pro
       flash vs qwen
       pro vs qwen

Outputs:
  embedding_data2_results/
    raw_outputs.csv
    case_group_output_counts.csv
    case_model_output_counts.csv

    case_level_group_pairwise_similarities.csv
    summary_group_pairwise_similarities.csv
    mean_group_similarity_matrix.csv

    case_level_model_similarities.csv
    summary_model_similarities.csv
    mean_model_similarity_matrix.csv

    low_similarity_cases_by_pair.csv
    avg_case_group_embeddings.npz
    avg_case_model_embeddings.npz
"""

import argparse
import itertools
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_ORDER = ["flash", "pro", "qwen"]


def clean_text(x):
    if x is None:
        return ""
    x = str(x).replace("\r", " ").replace("\n", " ").strip()
    x = " ".join(x.split())
    return x


def detect_model_from_group(group_name: str):
    """
    Map folder/group name to a model type.
    """
    g = group_name.lower()
    if "qwen" in g:
        return "qwen"
    if "flash" in g or "falsh" in g:
        return "flash"
    if "pro" in g:
        return "pro"
    return None


def extract_case_id(path: str):
    """
    Extract case id from names like case_15.json or case-15.json.
    """
    m = re.search(r"case[_\-]?(\d+)\.json$", path)
    if m:
        return int(m.group(1))
    return None


def safe_load_json_from_zip(zf, name):
    try:
        content = zf.read(name).decode("utf-8")
        return json.loads(content)
    except Exception as e:
        print(f"[WARN] Failed to read {name}: {e}")
        return None


def extract_text_from_obj(obj):
    """
    Extract diagnosis text from one sample.
    Prefer clean diagnosis fields; fall back to response-like fields.
    """
    if isinstance(obj, str):
        return clean_text(obj)

    if not isinstance(obj, dict):
        return ""

    priority_keys = [
        "diagnosis_text",
        "final_diagnosis",
        "diagnosis",
        "final_answer",
        "answer",
        "prediction",
        "doctor_answer",
        "doctor_response",
        "response",
        "output",
        "raw_response",
        "content",
        "text",
        "message",
    ]

    for key in priority_keys:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return clean_text(val)

    # Recursive fallback for nested structures
    for val in obj.values():
        if isinstance(val, dict):
            out = extract_text_from_obj(val)
            if out:
                return out
        elif isinstance(val, list):
            for item in val:
                out = extract_text_from_obj(item)
                if out:
                    return out

    return ""


def extract_reference(case_obj):
    if not isinstance(case_obj, dict):
        return ""

    keys = [
        "correct_diagnosis_reference",
        "correct_diagnosis",
        "gold_diagnosis",
        "reference_diagnosis",
        "ground_truth",
        "correct_answer",
        "gold_answer",
    ]

    for k in keys:
        v = case_obj.get(k)
        if isinstance(v, str) and v.strip():
            return clean_text(v)

    return ""


def get_samples(case_obj):
    """
    Find the list of model outputs inside one case json.
    """
    if isinstance(case_obj, dict):
        for key in ["samples", "outputs", "results", "runs"]:
            val = case_obj.get(key)
            if isinstance(val, list):
                return val
    if isinstance(case_obj, list):
        return case_obj
    return [case_obj]


def list_case_json_files(data_zip):
    """
    Return valid case json files as tuples:
      (zip_path, group_name, model_name, case_id)
    """
    files = []

    with zipfile.ZipFile(data_zip, "r") as zf:
        for name in zf.namelist():
            if "__MACOSX" in name:
                continue
            if os.path.basename(name).startswith("._"):
                continue
            if not name.endswith(".json"):
                continue

            case_id = extract_case_id(name)
            if case_id is None:
                continue

            parts = name.split("/")
            if len(parts) < 2:
                continue

            group = parts[-2]
            model = detect_model_from_group(group)
            if model is None:
                continue

            files.append((name, group, model, case_id))

    return sorted(files, key=lambda x: (x[3], x[1], x[0]))


def load_raw_outputs(data_zip, case_ids=None, case_start=None, case_end=None, max_outputs_per_group=10):
    """
    Load diagnosis outputs from zip into a dataframe.
    """
    rows = []
    json_files = list_case_json_files(data_zip)

    if not json_files:
        raise ValueError(
            "No valid JSON case files found. Folder names should include pro, flash/falsh, or qwen."
        )

    with zipfile.ZipFile(data_zip, "r") as zf:
        for name, group, model, case_id in json_files:
            if case_ids is not None and case_id not in case_ids:
                continue
            if case_ids is None:
                if case_start is not None and case_id < case_start:
                    continue
                if case_end is not None and case_id > case_end:
                    continue

            case_obj = safe_load_json_from_zip(zf, name)
            if case_obj is None:
                continue

            reference = extract_reference(case_obj)
            samples = get_samples(case_obj)

            if not isinstance(samples, list):
                continue

            samples = samples[:max_outputs_per_group]

            for idx, sample in enumerate(samples):
                diagnosis_text = extract_text_from_obj(sample)
                if not diagnosis_text:
                    continue

                run_id = sample.get("run", idx) if isinstance(sample, dict) else idx

                rows.append({
                    "model": model,
                    "group": group,
                    "case_id": case_id,
                    "run_id": run_id,
                    "run_order": idx + 1,
                    "source_file": name,
                    "correct_diagnosis_reference": reference,
                    "diagnosis_text": diagnosis_text,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No diagnosis outputs extracted from Data2.zip.")

    return df.sort_values(["case_id", "model", "group", "run_order"]).reset_index(drop=True)


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def average_embedding(embed_model, texts, batch_size=32):
    """
    Embed all texts, average them, and normalize the averaged vector.
    """
    if len(texts) == 0:
        return None

    embs = embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    avg = np.mean(embs, axis=0)
    return normalize_vector(avg)


def cosine(a, b):
    if a is None or b is None:
        return np.nan
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def summarize_pairwise(case_df, pair_cols):
    rows = []
    for col in pair_cols:
        values = case_df[col].dropna()
        rows.append({
            "comparison": col,
            "mean_case_level_similarity": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "n_cases": int(values.shape[0]),
            "mean_embedding_gap": 1 - values.mean(),
        })
    return pd.DataFrame(rows)


def build_mean_matrix(case_df, items, pair_cols):
    matrix = pd.DataFrame(index=items, columns=items, dtype=float)

    for item in items:
        matrix.loc[item, item] = 1.0

    for a, b in itertools.combinations(items, 2):
        c1 = f"{a}_vs_{b}"
        c2 = f"{b}_vs_{a}"
        if c1 in case_df.columns:
            col = c1
        elif c2 in case_df.columns:
            col = c2
        else:
            continue

        val = case_df[col].mean()
        matrix.loc[a, b] = val
        matrix.loc[b, a] = val

    return matrix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_zip",
        type=str,
        required=True,
        help="Path to Data2.zip",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer embedding model name or local path",
    )
    parser.add_argument(
        "--case_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit anchor case ids, e.g. --case_ids 2 5 15 18 23 24 25 26 28",
    )
    parser.add_argument(
        "--case_start",
        type=int,
        default=None,
        help="Optional start case id",
    )
    parser.add_argument(
        "--case_end",
        type=int,
        default=None,
        help="Optional end case id",
    )
    parser.add_argument(
        "--max_outputs_per_group",
        type=int,
        default=10,
        help="Use first N outputs per group/case file. Default 10.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embedding_data2_results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading raw diagnosis outputs...")
    raw_df = load_raw_outputs(
        data_zip=args.data_zip,
        case_ids=set(args.case_ids) if args.case_ids else None,
        case_start=args.case_start,
        case_end=args.case_end,
        max_outputs_per_group=args.max_outputs_per_group,
    )
    raw_df.to_csv(output_dir / "raw_outputs.csv", index=False)

    print("\nDetected groups:")
    group_map = raw_df[["group", "model"]].drop_duplicates().sort_values(["model", "group"])
    print(group_map.to_string(index=False))

    case_group_counts = (
        raw_df.groupby(["case_id", "group"])
        .size()
        .reset_index(name="n_outputs")
        .pivot(index="case_id", columns="group", values="n_outputs")
        .reset_index()
    )
    case_group_counts.to_csv(output_dir / "case_group_output_counts.csv", index=False)

    case_model_counts = (
        raw_df.groupby(["case_id", "model"])
        .size()
        .reset_index(name="n_outputs")
        .pivot(index="case_id", columns="model", values="n_outputs")
        .reset_index()
    )
    case_model_counts.to_csv(output_dir / "case_model_output_counts.csv", index=False)

    print("\nOutput counts by case/group:")
    print(case_group_counts.to_string(index=False))

    all_groups = sorted(raw_df["group"].unique().tolist())
    available_models = [m for m in MODEL_ORDER if m in set(raw_df["model"])]
    if len(available_models) < 2:
        raise ValueError("Need at least two model types among flash/pro/qwen.")

    # Use common cases across all available models.
    model_case_sets = [set(raw_df[raw_df["model"] == m]["case_id"].unique()) for m in available_models]
    common_cases = sorted(set.intersection(*model_case_sets))
    if not common_cases:
        raise ValueError("No common case ids across available models.")

    print(f"\nCommon cases across {available_models}: {common_cases}")
    print(f"Number of common cases: {len(common_cases)}")

    print("\n[2/6] Loading embedding model...")
    embed_model = SentenceTransformer(args.model)

    print("[3/6] Computing group-level average embeddings...")
    group_embedding_store = {}
    group_meta_rows = []

    for case_id in common_cases:
        for group in all_groups:
            sub = raw_df[(raw_df["case_id"] == case_id) & (raw_df["group"] == group)]
            texts = sub["diagnosis_text"].tolist()
            avg_emb = average_embedding(embed_model, texts, batch_size=args.batch_size)

            key = f"case_{case_id}__{group}"
            group_embedding_store[key] = avg_emb

            ref = ""
            refs = sub["correct_diagnosis_reference"].dropna().astype(str).tolist()
            refs = [r for r in refs if r.strip()]
            if refs:
                ref = refs[0]

            model_name = sub["model"].iloc[0] if len(sub) > 0 else detect_model_from_group(group)

            group_meta_rows.append({
                "case_id": case_id,
                "group": group,
                "model": model_name,
                "n_outputs": len(texts),
                "correct_diagnosis_reference": ref,
            })

    pd.DataFrame(group_meta_rows).to_csv(output_dir / "case_group_embedding_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / "avg_case_group_embeddings.npz",
        **{k: v for k, v in group_embedding_store.items() if v is not None},
    )

    print("[4/6] Computing model-level average embeddings...")
    model_embedding_store = {}
    model_meta_rows = []

    for case_id in common_cases:
        for model_name in available_models:
            sub = raw_df[(raw_df["case_id"] == case_id) & (raw_df["model"] == model_name)]
            texts = sub["diagnosis_text"].tolist()
            avg_emb = average_embedding(embed_model, texts, batch_size=args.batch_size)

            key = f"case_{case_id}__{model_name}"
            model_embedding_store[key] = avg_emb

            ref = ""
            refs = sub["correct_diagnosis_reference"].dropna().astype(str).tolist()
            refs = [r for r in refs if r.strip()]
            if refs:
                ref = refs[0]

            model_meta_rows.append({
                "case_id": case_id,
                "model": model_name,
                "n_outputs": len(texts),
                "correct_diagnosis_reference": ref,
            })

    pd.DataFrame(model_meta_rows).to_csv(output_dir / "case_model_embedding_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / "avg_case_model_embeddings.npz",
        **{k: v for k, v in model_embedding_store.items() if v is not None},
    )

    print("[5/6] Computing group-level pairwise similarities...")
    group_pairs = list(itertools.combinations(all_groups, 2))
    group_case_rows = []

    for case_id in common_cases:
        row = {"case_id": case_id}
        refs = raw_df[
            (raw_df["case_id"] == case_id)
            & (raw_df["correct_diagnosis_reference"].astype(str).str.len() > 0)
        ]["correct_diagnosis_reference"].tolist()
        row["correct_diagnosis_reference"] = refs[0] if refs else ""

        for g1, g2 in group_pairs:
            v1 = group_embedding_store.get(f"case_{case_id}__{g1}")
            v2 = group_embedding_store.get(f"case_{case_id}__{g2}")
            col = f"{g1}_vs_{g2}"
            row[col] = cosine(v1, v2)
            row[f"{col}_gap"] = 1 - row[col] if not pd.isna(row[col]) else np.nan

        group_case_rows.append(row)

    group_case_df = pd.DataFrame(group_case_rows)
    group_case_df.to_csv(output_dir / "case_level_group_pairwise_similarities.csv", index=False)

    group_pair_cols = [f"{g1}_vs_{g2}" for g1, g2 in group_pairs]
    group_summary_df = summarize_pairwise(group_case_df, group_pair_cols)
    group_summary_df.to_csv(output_dir / "summary_group_pairwise_similarities.csv", index=False)

    group_matrix = build_mean_matrix(group_case_df, all_groups, group_pair_cols)
    group_matrix.to_csv(output_dir / "mean_group_similarity_matrix.csv")

    print("[6/6] Computing model-level pairwise similarities...")
    model_pairs = list(itertools.combinations(available_models, 2))
    model_case_rows = []

    for case_id in common_cases:
        row = {"case_id": case_id}
        refs = raw_df[
            (raw_df["case_id"] == case_id)
            & (raw_df["correct_diagnosis_reference"].astype(str).str.len() > 0)
        ]["correct_diagnosis_reference"].tolist()
        row["correct_diagnosis_reference"] = refs[0] if refs else ""

        for m1, m2 in model_pairs:
            v1 = model_embedding_store.get(f"case_{case_id}__{m1}")
            v2 = model_embedding_store.get(f"case_{case_id}__{m2}")
            col = f"{m1}_vs_{m2}"
            row[col] = cosine(v1, v2)
            row[f"{col}_gap"] = 1 - row[col] if not pd.isna(row[col]) else np.nan

        pair_cols = [f"{m1}_vs_{m2}" for m1, m2 in model_pairs]
        row["average_pairwise_similarity"] = np.nanmean([row[c] for c in pair_cols])
        row["average_pairwise_gap"] = 1 - row["average_pairwise_similarity"]

        model_case_rows.append(row)

    model_case_df = pd.DataFrame(model_case_rows)
    model_case_df.to_csv(output_dir / "case_level_model_similarities.csv", index=False)

    model_pair_cols = [f"{m1}_vs_{m2}" for m1, m2 in model_pairs]
    model_summary_df = summarize_pairwise(
        model_case_df,
        model_pair_cols + ["average_pairwise_similarity"],
    )
    model_summary_df.to_csv(output_dir / "summary_model_similarities.csv", index=False)

    model_matrix = build_mean_matrix(model_case_df, available_models, model_pair_cols)
    model_matrix.to_csv(output_dir / "mean_model_similarity_matrix.csv")

    # Low similarity ranking for every model pair and for average similarity.
    low_rows = []
    for col in model_pair_cols:
        tmp = model_case_df[["case_id", "correct_diagnosis_reference", col]].copy()
        tmp = tmp.rename(columns={col: "similarity"})
        tmp["comparison"] = col
        tmp["embedding_gap"] = 1 - tmp["similarity"]
        low_rows.append(tmp)

    tmp_avg = model_case_df[["case_id", "correct_diagnosis_reference", "average_pairwise_similarity"]].copy()
    tmp_avg = tmp_avg.rename(columns={"average_pairwise_similarity": "similarity"})
    tmp_avg["comparison"] = "average_pairwise_similarity"
    tmp_avg["embedding_gap"] = 1 - tmp_avg["similarity"]
    low_rows.append(tmp_avg)

    low_df = pd.concat(low_rows, ignore_index=True)
    low_df = low_df.sort_values(["comparison", "similarity"])
    low_df.to_csv(output_dir / "low_similarity_cases_by_pair.csv", index=False)

    print("\nDone.")
    print(f"Saved all results to: {output_dir}")

    print("\nModel-level summary:")
    print(model_summary_df.to_string(index=False))

    print("\nModel-level mean similarity matrix:")
    print(model_matrix)

    print("\nLowest average-pairwise cases:")
    print(
        model_case_df.sort_values("average_pairwise_similarity")
        .head(10)[[
            "case_id",
            "correct_diagnosis_reference",
            "average_pairwise_similarity",
            "average_pairwise_gap",
        ]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
