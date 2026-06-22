#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case-level embedding similarity (诊断漂移 · Method B).

把每个 case 在不同模型组下的输出 embedding 成向量、按 case 取平均并归一化，
再算两两 cosine 相似度，用来衡量模型/版本之间的诊断漂移。

统一了原来的三个脚本（full_dialogue / analyze_data2 / gpt），区别只在于
「嵌入哪段文本」——现在由 --text_field 决定，两个实验并列对等：

  --text_field diagnosis_text   只比最终诊断（结论层面）
  --text_field full_dialogue    比整段问诊对话（过程层面）

输入是目录（不再是 zip），目录下形如：
    <data_dir>/deepseek_flash_1/case_2.json
    <data_dir>/deepseek_pro_2/case_5.json
    <data_dir>/gpt_5_5_1/case_2.json
每个 case_*.json 是 generate_diagnosis_distribution 的产出（含 samples / full_dialogue / diagnosis_text）。

模型组从文件夹名自动推断（去掉尾部的 _<run> 批次号），无需任何硬编码模型表：
    deepseek_flash_1 / deepseek_flash_2 -> deepseek_flash
    gpt_5_5_1        / gpt_5_5_2        -> gpt_5_5
"""

import argparse
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_FIELD_CHOICES = ["diagnosis_text", "full_dialogue"]


def clean_text(x):
    if x is None:
        return ""
    x = str(x).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(x.split())


def model_from_group(group_name: str) -> str:
    """模型家族 = 文件夹名去掉尾部的批次号 _<digits>。

    deepseek_flash_1 -> deepseek_flash ; gpt_5_4_mini_2 -> gpt_5_4_mini
    没有尾号时原样返回。完全由数据驱动，新增模型无需改代码。
    """
    return re.sub(r"_\d+$", "", group_name)


def extract_case_id(path: str):
    m = re.search(r"case[_\-]?(\d+)\.json$", path)
    return int(m.group(1)) if m else None


def safe_load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
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
    """整段对话文本；缺失时回退到 dialogue-like 字段。"""
    if isinstance(sample, str):
        return clean_text(sample)
    if not isinstance(sample, dict):
        return ""

    val = sample.get("full_dialogue")
    if isinstance(val, str) and val.strip():
        return clean_text(val)

    for key in ["dialogue", "conversation", "full_conversation", "transcript", "messages"]:
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


def extract_diagnosis_text(obj):
    """最终诊断文本；按诊断字段优先，再递归兜底（与原 gpt/data2 脚本一致）。"""
    if isinstance(obj, str):
        return clean_text(obj)
    if not isinstance(obj, dict):
        return ""

    priority_keys = [
        "diagnosis_text", "final_diagnosis", "diagnosis", "final_answer",
        "answer", "prediction", "doctor_answer", "doctor_response",
        "response", "output", "raw_response", "content", "text", "message",
    ]
    for key in priority_keys:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return clean_text(val)

    for val in obj.values():
        if isinstance(val, dict):
            out = extract_diagnosis_text(val)
            if out:
                return out
        elif isinstance(val, list):
            for item in val:
                out = extract_diagnosis_text(item)
                if out:
                    return out
    return ""


def iter_case_json_files(data_dirs):
    """遍历一个或多个数据目录，产出 (path, group, case_id)。

    group = case_*.json 的父目录名；自动跳过 macOS 的 ._ / .DS_Store 残留。
    """
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [data_dirs]
    for data_dir in data_dirs:
        root = Path(data_dir)
        for path in sorted(root.rglob("*.json")):
            if path.name.startswith("._") or path.name == ".DS_Store":
                continue
            case_id = extract_case_id(path.name)
            if case_id is None:
                continue
            group = path.parent.name
            yield path, group, case_id


def load_raw(data_dirs, text_field, case_ids=None, case_start=None, case_end=None,
             max_outputs_per_group=10):
    extract_embed_text = (
        extract_full_dialogue if text_field == "full_dialogue" else extract_diagnosis_text
    )

    rows = []
    group_model_map = {}

    for path, group, case_id in iter_case_json_files(data_dirs):
        if case_ids is not None and case_id not in case_ids:
            continue
        if case_ids is None:
            if case_start is not None and case_id < case_start:
                continue
            if case_end is not None and case_id > case_end:
                continue

        case_obj = safe_load_json(path)
        if case_obj is None:
            continue

        model = model_from_group(group)
        group_model_map[group] = model
        reference = extract_reference(case_obj)
        samples = get_samples(case_obj)
        if not isinstance(samples, list):
            continue
        samples = samples[:max_outputs_per_group]

        for idx, sample in enumerate(samples):
            embed_text = extract_embed_text(sample)
            if not embed_text:
                continue
            run_id = sample.get("run", idx) if isinstance(sample, dict) else idx
            rows.append({
                "model": model,
                "group": group,
                "case_id": case_id,
                "run_id": run_id,
                "run_order": idx + 1,
                "source_file": str(path),
                "correct_diagnosis_reference": reference,
                "embedding_text": embed_text,
                "embedding_text_char_len": len(embed_text),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No '{text_field}' texts extracted from {data_dirs}. "
            "Expected <data_dir>/<group>/case_*.json files."
        )

    print("Detected group -> model mapping:")
    for g, m in sorted(group_model_map.items()):
        print(f"  {g} -> {m}")

    return df.sort_values(["case_id", "group", "run_order"]).reset_index(drop=True)


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def average_embedding(embed_model, texts, batch_size=8):
    if len(texts) == 0:
        return None
    embs = embed_model.encode(
        texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
    )
    return normalize_vector(np.mean(embs, axis=0))


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
        col1, col2 = f"{a}_vs_{b}", f"{b}_vs_{a}"
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
    low_df = pd.concat(rows, ignore_index=True).sort_values(["comparison", "similarity"])
    low_df.to_csv(output_path, index=False)
    return low_df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, nargs="+", required=True,
                        help="一个或多个数据目录，下含 <group>/case_*.json")
    parser.add_argument("--text_field", required=True, choices=TEXT_FIELD_CHOICES,
                        help="嵌入哪段文本：diagnosis_text(只比最终诊断) 或 "
                             "full_dialogue(比整段问诊)。两个实验并列，必须显式指定。")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="SentenceTransformer 模型名或本地路径")
    parser.add_argument("--case_ids", type=int, nargs="*", default=None)
    parser.add_argument("--case_start", type=int, default=None)
    parser.add_argument("--case_end", type=int, default=None)
    parser.add_argument("--max_outputs_per_group", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="默认 results/embedding_similarity/<text_field>")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "results", "embedding_similarity", args.text_field)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading '{args.text_field}' outputs...")
    raw_df = load_raw(
        data_dirs=args.data_dir,
        text_field=args.text_field,
        case_ids=set(args.case_ids) if args.case_ids else None,
        case_start=args.case_start,
        case_end=args.case_end,
        max_outputs_per_group=args.max_outputs_per_group,
    )
    raw_df.to_csv(output_dir / "raw_outputs.csv", index=False)

    count_df = (
        raw_df.groupby(["case_id", "group"]).size()
        .reset_index(name="n_outputs")
        .pivot(index="case_id", columns="group", values="n_outputs")
        .reset_index()
    )
    count_df.to_csv(output_dir / "case_group_output_counts.csv", index=False)
    print("\nOutput counts by case/group:")
    print(count_df.to_string(index=False))

    print("\n[2/6] Loading embedding model...")
    embed_model = SentenceTransformer(args.model)

    groups = sorted(raw_df["group"].unique().tolist())
    group_pairs = list(itertools.combinations(groups, 2))
    case_sets = [set(raw_df[raw_df["group"] == g]["case_id"].unique()) for g in groups]
    common_cases = sorted(set.intersection(*case_sets)) if case_sets else []
    print(f"\nGroups: {groups}")
    print(f"Common cases across all groups: {common_cases} (n={len(common_cases)})")

    print("\n[3/6] Computing group-level average embeddings by case...")
    group_emb_store = {}
    group_meta_rows = []
    for case_id in common_cases:
        for group in groups:
            sub = raw_df[(raw_df["case_id"] == case_id) & (raw_df["group"] == group)]
            texts = sub["embedding_text"].tolist()
            group_emb_store[f"case_{case_id}__{group}"] = average_embedding(
                embed_model, texts, batch_size=args.batch_size)
            refs = [r for r in sub["correct_diagnosis_reference"].dropna().astype(str).tolist() if r.strip()]
            group_meta_rows.append({
                "case_id": case_id,
                "group": group,
                "model": sub["model"].iloc[0] if len(sub) else "",
                "n_outputs": len(texts),
                "correct_diagnosis_reference": refs[0] if refs else "",
                "mean_embedding_text_char_len": np.mean([len(t) for t in texts]) if texts else np.nan,
            })
    pd.DataFrame(group_meta_rows).to_csv(output_dir / "case_group_embedding_metadata.csv", index=False)
    np.savez_compressed(output_dir / "avg_case_group_embeddings.npz",
                        **{k: v for k, v in group_emb_store.items() if v is not None})

    print("[4/6] Computing group-level pairwise similarities...")
    group_case_rows = []
    for case_id in common_cases:
        row = {"case_id": case_id}
        refs = raw_df[(raw_df["case_id"] == case_id)
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
    summarize_pairwise(group_case_df, group_pair_cols).to_csv(
        output_dir / "summary_group_pairwise_similarities.csv", index=False)
    group_matrix = build_matrix(groups, group_case_df, group_pair_cols)
    group_matrix.to_csv(output_dir / "mean_group_similarity_matrix.csv")
    export_low_similarity(group_case_df, group_pair_cols,
                          output_dir / "low_similarity_group_cases_by_pair.csv")

    print("[5/6] Computing merged model-level embeddings by case...")
    available_models = sorted(set(raw_df["model"]))
    model_pairs = list(itertools.combinations(available_models, 2))
    model_emb_store = {}
    model_meta_rows = []
    for case_id in common_cases:
        for model_name in available_models:
            sub = raw_df[(raw_df["case_id"] == case_id) & (raw_df["model"] == model_name)]
            texts = sub["embedding_text"].tolist()
            model_emb_store[f"case_{case_id}__{model_name}"] = average_embedding(
                embed_model, texts, batch_size=args.batch_size)
            refs = [r for r in sub["correct_diagnosis_reference"].dropna().astype(str).tolist() if r.strip()]
            model_meta_rows.append({
                "case_id": case_id,
                "model": model_name,
                "n_outputs": len(texts),
                "correct_diagnosis_reference": refs[0] if refs else "",
                "mean_embedding_text_char_len": np.mean([len(t) for t in texts]) if texts else np.nan,
            })
    pd.DataFrame(model_meta_rows).to_csv(output_dir / "case_model_embedding_metadata.csv", index=False)
    np.savez_compressed(output_dir / "avg_case_model_embeddings.npz",
                        **{k: v for k, v in model_emb_store.items() if v is not None})

    print("[6/6] Computing model-level pairwise similarities...")
    model_case_rows = []
    for case_id in common_cases:
        row = {"case_id": case_id}
        refs = raw_df[(raw_df["case_id"] == case_id)
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
        if pair_cols:
            row["average_pairwise_similarity"] = np.nanmean([row[c] for c in pair_cols])
            row["average_pairwise_gap"] = 1 - row["average_pairwise_similarity"]
        model_case_rows.append(row)
    model_case_df = pd.DataFrame(model_case_rows)
    model_case_df.to_csv(output_dir / "case_level_model_similarities.csv", index=False)

    model_pair_cols = [f"{m1}_vs_{m2}" for m1, m2 in model_pairs]
    summarize_pairwise(model_case_df, model_pair_cols + (["average_pairwise_similarity"] if model_pair_cols else [])).to_csv(
        output_dir / "summary_model_similarities.csv", index=False)
    build_matrix(available_models, model_case_df, model_pair_cols).to_csv(
        output_dir / "mean_model_similarity_matrix.csv")
    if model_pair_cols:
        export_low_similarity(model_case_df, model_pair_cols + ["average_pairwise_similarity"],
                              output_dir / "low_similarity_model_cases_by_pair.csv")

    print(f"\nDone. Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
