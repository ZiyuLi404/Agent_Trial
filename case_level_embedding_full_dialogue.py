#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case-level embedding comparison using full_dialogue instead of diagnosis_text.

Input:
  Data.zip with folders such as:
    Data/deepseek_flash_1/case_2.json
    Data/deepseek_flash_2/case_2.json
    Data/deepseek_pro_1/case_2.json
    Data/deepseek_pro_2/case_2.json
    Data/Qwen_plus_turbo_1/case_2.json
    Data/Qwen_plus_turbo_2/case_2.json

Method:
  For each case and each dataset/group:
    1. Read samples.
    2. Use full_dialogue from the first N samples, default N=10.
    3. Embed each full_dialogue.
    4. Average embeddings within the same case and group.
    5. Normalize the averaged embedding.

  Then:
    - Compute all pairwise cosine similarities among groups for each case.
    - Average each pairwise similarity across cases.

  Also:
    - Merge groups into model-level categories:
        flash = folders containing "flash" or "falsh"
        pro   = folders containing "pro"
        qwen  = folders containing "qwen"
    - Compute flash/pro/qwen pairwise similarities using full_dialogue.

Outputs:
  embedding_full_dialogue_results/
    raw_full_dialogue_outputs.csv
    case_group_output_counts.csv
    case_level_group_pairwise_similarities.csv
    summary_group_pairwise_similarities.csv
    mean_group_similarity_matrix.csv
    case_level_model_similarities.csv
    summary_model_similarities.csv
    mean_model_similarity_matrix.csv
    low_similarity_group_cases_by_pair.csv
    low_similarity_model_cases_by_pair.csv
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


MODEL_ORDER = ["flash", "pro", "qwen", "gpt_5_4_mini", "gpt_5_5"]


def clean_text(x):
    if x is None:
        return ""
    x = str(x).replace("\r", " ").replace("\n", " ").strip()
    x = " ".join(x.split())
    return x


def detect_model_from_group(group_name: str):
    g = group_name.lower()
    if "qwen" in g:
        return "qwen"
    if "flash" in g or "falsh" in g:
        return "flash"
    if "pro" in g:
        return "pro"
    if "gpt" in g:
        # Collapse repeat-batch suffix into a model family label:
        #   gpt_5_5_1 / gpt_5_5_2      -> gpt_5_5
        #   gpt_5_4_mini_1 / ..._2     -> gpt_5_4_mini
        return re.sub(r"_\d+$", "", g)
    return None


def extract_case_id(path: str):
    m = re.search(r"case[_\-]?(\d+)\.json$", path)
    if m:
        return int(m.group(1))
    return None


def safe_load_json_from_zip(zf, name):
    try:
        return json.loads(zf.read(name).decode("utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read {name}: {e}")
        return None


def extract_reference(case_obj):
    if not isinstance(case_obj, dict):
        return ""
    for k in [
        "correct_diagnosis_reference",
        "correct_diagnosis",
        "gold_diagnosis",
        "reference_diagnosis",
        "ground_truth",
        "correct_answer",
        "gold_answer",
    ]:
        v = case_obj.get(k)
        if isinstance(v, str) and v.strip():
            return clean_text(v)
    return ""


def get_samples(case_obj):
    if isinstance(case_obj, dict):
        for key in ["samples", "outputs", "results", "runs"]:
            val = case_obj.get(key)
            if isinstance(val, list):
                return val
    if isinstance(case_obj, list):
        return case_obj
    return [case_obj]


def extract_full_dialogue(sample):
    """
    The embedding text is full_dialogue.
    If a sample does not have full_dialogue, fallback to dialogue-like fields.
    """
    if isinstance(sample, str):
        return clean_text(sample)

    if not isinstance(sample, dict):
        return ""

    # Primary field requested by user
    val = sample.get("full_dialogue")
    if isinstance(val, str) and val.strip():
        return clean_text(val)

    # Fallbacks in case a file uses slightly different names
    fallback_keys = [
        "dialogue",
        "conversation",
        "full_conversation",
        "transcript",
        "messages",
    ]
    for key in fallback_keys:
        val = sample.get(key)
        if isinstance(val, str) and val.strip():
            return clean_text(val)
        if isinstance(val, list):
            parts = []
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


def load_raw_full_dialogues(data_zips, case_ids=None, case_start=None, case_end=None, max_outputs_per_group=10):
    # Accept either a single zip path or a list of zip paths (e.g. Data.zip + Data_gpt.zip),
    # so multiple model families can be analyzed together in one run.
    if isinstance(data_zips, (str, os.PathLike)):
        data_zips = [data_zips]

    rows = []
    group_model_map = {}

    for data_zip in data_zips:
        with zipfile.ZipFile(data_zip, "r") as zf:
            names = zf.namelist()
            json_files = []

            for name in names:
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

                group_model_map[group] = model
                json_files.append((name, group, model, case_id))

            for name, group, model, case_id in sorted(json_files):
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
                    full_dialogue = extract_full_dialogue(sample)
                    if not full_dialogue:
                        continue

                    diagnosis_text = ""
                    if isinstance(sample, dict):
                        diagnosis_text = clean_text(
                            sample.get("diagnosis_text")
                            or sample.get("final_diagnosis")
                            or sample.get("diagnosis")
                            or sample.get("answer")
                            or sample.get("raw_response")
                            or ""
                        )
                        run_id = sample.get("run", idx)
                    else:
                        run_id = idx

                    rows.append({
                        "model": model,
                        "group": group,
                        "case_id": case_id,
                        "run_id": run_id,
                        "run_order": idx + 1,
                        "source_file": name,
                        "correct_diagnosis_reference": reference,
                        "diagnosis_text_for_reference_only": diagnosis_text,
                        "full_dialogue": full_dialogue,
                        "full_dialogue_char_len": len(full_dialogue),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "No full_dialogue texts extracted. Folder names should include "
            "pro, flash/falsh, qwen, or gpt."
        )

    print("Detected group -> model mapping:")
    for g, m in sorted(group_model_map.items()):
        print(f"  {g} -> {m}")

    return df.sort_values(["case_id", "group", "run_order"]).reset_index(drop=True)


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def average_embedding(embed_model, texts, batch_size=8):
    """
    Embed full dialogues, average them, then normalize the average vector.
    full_dialogue can be long, so batch_size default is smaller.
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


def build_matrix(labels, case_df, pair_cols):
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for x in labels:
        matrix.loc[x, x] = 1.0

    for a, b in itertools.combinations(labels, 2):
        col1 = f"{a}_vs_{b}"
        col2 = f"{b}_vs_{a}"
        col = col1 if col1 in case_df.columns else col2
        if col in case_df.columns:
            val = case_df[col].mean()
            matrix.loc[a, b] = val
            matrix.loc[b, a] = val

    return matrix


def export_low_similarity(case_df, pair_cols, output_path):
    rows = []
    for col in pair_cols:
        tmp = case_df[["case_id", "correct_diagnosis_reference", col]].copy()
        tmp = tmp.rename(columns={col: "similarity"})
        tmp["comparison"] = col
        tmp["embedding_gap"] = 1 - tmp["similarity"]
        rows.append(tmp)
    low_df = pd.concat(rows, ignore_index=True)
    low_df = low_df.sort_values(["comparison", "similarity"])
    low_df.to_csv(output_path, index=False)
    return low_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_zip", type=str, nargs="+", required=True, help="Path(s) to one or more data zips, e.g. --data_zip Data.zip Data_gpt.zip")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="SentenceTransformer model name or local path")
    parser.add_argument("--case_ids", type=int, nargs="*", default=None, help="Optional explicit case ids, e.g. --case_ids 2 5 15 18 23 24 25 26 28")
    parser.add_argument("--case_start", type=int, default=None)
    parser.add_argument("--case_end", type=int, default=None)
    parser.add_argument("--max_outputs_per_group", type=int, default=10, help="Use first N full_dialogue samples per group/case file")
    parser.add_argument("--output_dir", type=str, default="embedding_full_dialogue_results")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading full_dialogue outputs...")
    raw_df = load_raw_full_dialogues(
        data_zips=args.data_zip,
        case_ids=set(args.case_ids) if args.case_ids else None,
        case_start=args.case_start,
        case_end=args.case_end,
        max_outputs_per_group=args.max_outputs_per_group,
    )
    raw_df.to_csv(output_dir / "raw_full_dialogue_outputs.csv", index=False)

    count_df = (
        raw_df.groupby(["case_id", "group"])
        .size()
        .reset_index(name="n_outputs")
        .pivot(index="case_id", columns="group", values="n_outputs")
        .reset_index()
    )
    count_df.to_csv(output_dir / "case_group_output_counts.csv", index=False)

    print("\nOutput counts by case/group:")
    print(count_df.to_string(index=False))

    print("\n[2/6] Loading embedding model...")
    embed_model = SentenceTransformer(args.model)

    # Common cases and groups
    groups = sorted(raw_df["group"].unique().tolist())
    group_pairs = list(itertools.combinations(groups, 2))

    case_sets = []
    for g in groups:
        case_sets.append(set(raw_df[raw_df["group"] == g]["case_id"].unique()))
    common_cases = sorted(set.intersection(*case_sets))

    print(f"\nGroups: {groups}")
    print(f"Common cases across all groups: {common_cases}")
    print(f"Number of common cases: {len(common_cases)}")

    print("\n[3/6] Computing group-level average embeddings by case...")
    group_emb_store = {}
    group_meta_rows = []

    for case_id in common_cases:
        for group in groups:
            sub = raw_df[(raw_df["case_id"] == case_id) & (raw_df["group"] == group)]
            texts = sub["full_dialogue"].tolist()
            avg_emb = average_embedding(embed_model, texts, batch_size=args.batch_size)
            key = f"case_{case_id}__{group}"
            group_emb_store[key] = avg_emb

            ref = ""
            refs = sub["correct_diagnosis_reference"].dropna().astype(str).tolist()
            refs = [r for r in refs if r.strip()]
            if refs:
                ref = refs[0]

            group_meta_rows.append({
                "case_id": case_id,
                "group": group,
                "model": sub["model"].iloc[0] if len(sub) else "",
                "n_outputs": len(texts),
                "correct_diagnosis_reference": ref,
                "mean_full_dialogue_char_len": np.mean([len(t) for t in texts]) if texts else np.nan,
            })

    pd.DataFrame(group_meta_rows).to_csv(output_dir / "case_group_embedding_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / "avg_case_group_embeddings.npz",
        **{k: v for k, v in group_emb_store.items() if v is not None}
    )

    print("[4/6] Computing group-level pairwise similarities...")
    group_case_rows = []
    for case_id in common_cases:
        row = {"case_id": case_id}
        refs = raw_df[
            (raw_df["case_id"] == case_id)
            & (raw_df["correct_diagnosis_reference"].astype(str).str.len() > 0)
        ]["correct_diagnosis_reference"].tolist()
        row["correct_diagnosis_reference"] = refs[0] if refs else ""

        for g1, g2 in group_pairs:
            v1 = group_emb_store.get(f"case_{case_id}__{g1}")
            v2 = group_emb_store.get(f"case_{case_id}__{g2}")
            col = f"{g1}_vs_{g2}"
            row[col] = cosine(v1, v2)
            row[f"{col}_gap"] = 1 - row[col] if not pd.isna(row[col]) else np.nan

        group_case_rows.append(row)

    group_case_df = pd.DataFrame(group_case_rows)
    group_case_df.to_csv(output_dir / "case_level_group_pairwise_similarities.csv", index=False)

    group_pair_cols = [f"{g1}_vs_{g2}" for g1, g2 in group_pairs]
    group_summary_df = summarize_pairwise(group_case_df, group_pair_cols)
    group_summary_df.to_csv(output_dir / "summary_group_pairwise_similarities.csv", index=False)

    group_matrix = build_matrix(groups, group_case_df, group_pair_cols)
    group_matrix.to_csv(output_dir / "mean_group_similarity_matrix.csv")
    export_low_similarity(group_case_df, group_pair_cols, output_dir / "low_similarity_group_cases_by_pair.csv")

    print("[5/6] Computing merged model-level embeddings by case...")
    detected_models = set(raw_df["model"])
    # Use MODEL_ORDER as the preferred ordering, then append any other detected
    # model families (e.g. newly added gpt variants) so nothing is silently dropped.
    available_models = [m for m in MODEL_ORDER if m in detected_models]
    available_models += sorted(m for m in detected_models if m not in MODEL_ORDER)
    model_pairs = list(itertools.combinations(available_models, 2))

    model_emb_store = {}
    model_meta_rows = []

    for case_id in common_cases:
        for model_name in available_models:
            sub = raw_df[(raw_df["case_id"] == case_id) & (raw_df["model"] == model_name)]
            texts = sub["full_dialogue"].tolist()
            avg_emb = average_embedding(embed_model, texts, batch_size=args.batch_size)
            key = f"case_{case_id}__{model_name}"
            model_emb_store[key] = avg_emb

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
                "mean_full_dialogue_char_len": np.mean([len(t) for t in texts]) if texts else np.nan,
            })

    pd.DataFrame(model_meta_rows).to_csv(output_dir / "case_model_embedding_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / "avg_case_model_embeddings.npz",
        **{k: v for k, v in model_emb_store.items() if v is not None}
    )

    print("[6/6] Computing model-level pairwise similarities...")
    model_case_rows = []
    for case_id in common_cases:
        row = {"case_id": case_id}
        refs = raw_df[
            (raw_df["case_id"] == case_id)
            & (raw_df["correct_diagnosis_reference"].astype(str).str.len() > 0)
        ]["correct_diagnosis_reference"].tolist()
        row["correct_diagnosis_reference"] = refs[0] if refs else ""

        for m1, m2 in model_pairs:
            v1 = model_emb_store.get(f"case_{case_id}__{m1}")
            v2 = model_emb_store.get(f"case_{case_id}__{m2}")
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
    model_summary_df = summarize_pairwise(model_case_df, model_pair_cols + ["average_pairwise_similarity"])
    model_summary_df.to_csv(output_dir / "summary_model_similarities.csv", index=False)

    model_matrix = build_matrix(available_models, model_case_df, model_pair_cols)
    model_matrix.to_csv(output_dir / "mean_model_similarity_matrix.csv")
    export_low_similarity(model_case_df, model_pair_cols + ["average_pairwise_similarity"], output_dir / "low_similarity_model_cases_by_pair.csv")

    print("\nDone.")
    print(f"Saved outputs to: {output_dir}")

    print("\nGroup-level summary:")
    print(group_summary_df.to_string(index=False))

    print("\nGroup-level mean similarity matrix:")
    print(group_matrix)

    print("\nModel-level summary:")
    print(model_summary_df.to_string(index=False))

    print("\nModel-level mean similarity matrix:")
    print(model_matrix)


if __name__ == "__main__":
    main()
