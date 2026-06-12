#!/usr/bin/env python3
"""
ICD-10-CM Categorization and Version Drift Comparison Tool

Maps diagnosis_text from simulation result folders to ICD-10-CM codes via LLM,
recomputes distributions at the ICD-code level, then compares folders pairwise
using KL and JS divergence to detect LLM version/update drift.

Example commands
----------------
# Categorize one folder (cases 0-19, all 10 runs):
  python result_categorize/icd10_categorize_compare.py \
    --mode categorize \
    --folders 50case_10runs_flash \
    --cases 0-19 --runs 0-9 \
    --temperature temp_0.05 \
    --run_name analysis_v1

# Compare two folders (must have been categorized first, or use --mode both):
  python result_categorize/icd10_categorize_compare.py \
    --mode compare \
    --folders 50case_10runs_flash other_folder \
    --cases 0-19 \
    --temperature temp_0.05 \
    --run_name analysis_v1

# Categorize then compare in one step:
  python result_categorize/icd10_categorize_compare.py \
    --mode both \
    --folders 50case_10runs_flash other_folder \
    --cases 0-39 --runs 0-9 \
    --temperature temp_0.05 \
    --run_name analysis_v1 \
    --llm_model deepseek-chat

# Re-run compare only (skip LLM calls, reuse existing categorization):
  python result_categorize/icd10_categorize_compare.py \
    --mode both \
    --folders 50case_10runs_flash other_folder \
    --cases 0-39 \
    --temperature temp_0.05 \
    --run_name analysis_v1 \
    --reuse_existing_categorization

Required env vars:
  DEEPSEEK_API_KEY   → uses https://api.deepseek.com
  or OPENAI_API_KEY  → uses default OpenAI base URL
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICD-10-CM categorization and version drift comparison."
    )
    p.add_argument("--folders", nargs="+", required=True,
                   help="Result folder names under results/")
    p.add_argument("--cases", default="0-39",
                   help="Cases: '0-19', '0,1,5,8', or '0-4,7,9-11'")
    p.add_argument("--runs", default="0-9",
                   help="Runs to include: '0-9', '0-4', or '0,2,4,6,8'")
    p.add_argument("--temperature", default="temp_0.05",
                   help="Temperature subfolder, e.g. temp_0.05")
    p.add_argument("--mode", choices=["categorize", "compare", "both"],
                   default="both")
    p.add_argument("--run_name", required=True,
                   help="Label for this analysis run (used in output paths)")
    p.add_argument("--llm_model", default="deepseek-chat",
                   help="Model name passed to the OpenAI-compatible API")
    p.add_argument("--epsilon", type=float, default=1e-9,
                   help="Laplace smoothing for KL divergence")
    p.add_argument("--reuse_existing_categorization", action="store_true",
                   help="Load existing .icd10.json files instead of re-calling LLM")
    p.add_argument("--results_dir", default="results",
                   help="Root directory containing result folders")
    p.add_argument("--out_dir", default="result_categorize",
                   help="Root output directory")
    p.add_argument("--icd_dict",
                   default="result_categorize/icd10cm_2026.jsonl",
                   help="Path to ICD-10-CM JSONL dictionary")
    p.add_argument("--cache_file",
                   default="result_categorize/icd10_mapping_cache.json",
                   help="Path to LLM mapping cache (keyed by diagnosis_text)")
    p.add_argument("--api_sleep", type=float, default=0.3,
                   help="Seconds to sleep between LLM API calls")
    p.add_argument("--api_key", default=None,
                   help="DeepSeek/OpenAI API key (overrides DEEPSEEK_API_KEY env var)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ID-spec parser:  "0-9"  |  "0,1,5"  |  "0-4,7,9-11"
# ---------------------------------------------------------------------------

def parse_id_spec(spec: str) -> list:
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            if part:
                ids.append(int(part))
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# ICD-10-CM dictionary
# ---------------------------------------------------------------------------

def load_icd_dict(path: str) -> dict:
    """Return {code_no_dot_uppercase: description}."""
    d = {}
    if not os.path.exists(path):
        logging.warning(f"ICD dictionary not found: {path}. LLM will have no local hints.")
        return d
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            code = obj["code"].upper().replace(".", "").replace(" ", "")
            d[code] = obj["description"]
    logging.info(f"Loaded {len(d):,} ICD-10-CM entries from {path}")
    return d


def build_word_index(icd_dict: dict) -> dict:
    """Inverted index: word (len>=4) → set of ICD codes. Built once at startup."""
    idx: dict = defaultdict(set)
    for code, desc in icd_dict.items():
        for w in re.findall(r"[a-zA-Z]{4,}", desc):
            idx[w.lower()].add(code)
    return idx


def get_icd_hints(diagnosis_text: str, icd_dict: dict, word_index: dict,
                   n: int = 8) -> list:
    """Return up to n ICD code hint strings ranked by keyword overlap."""
    keywords = re.findall(r"[a-zA-Z]{4,}", diagnosis_text)
    keywords = [w.lower() for w in keywords[:6]]
    scores: dict = defaultdict(int)
    for kw in keywords:
        for code in word_index.get(kw, []):
            scores[code] += 1
    top = sorted(scores, key=lambda c: -scores[c])[:n]
    return [f"{c}: {icd_dict[c]}" for c in top]


# ---------------------------------------------------------------------------
# LLM cache
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible)
# ---------------------------------------------------------------------------

def make_llm_client(api_key: Optional[str] = None):
    try:
        from openai import OpenAI
    except ImportError:
        logging.error("openai package not found. Install with: pip install openai")
        sys.exit(1)
    key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        logging.error(
            "Provide an API key via --api_key or set DEEPSEEK_API_KEY / OPENAI_API_KEY."
        )
        sys.exit(1)
    # Use DeepSeek base URL when the key came from --api_key or DEEPSEEK_API_KEY
    use_deepseek = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if use_deepseek:
        return OpenAI(api_key=key, base_url="https://api.deepseek.com")
    return OpenAI(api_key=key)


_SYSTEM_PROMPT = """\
You are a clinical coding specialist. Map the given diagnosis text to the most
specific and accurate ICD-10-CM code from 2026 edition.

Return ONLY a JSON object with exactly these fields:
{
  "icd10_code": "XXXXX",
  "icd10_code_dotted": "XXX.XX",
  "label": "Official ICD-10-CM description",
  "confidence": 0.0,
  "reason": "brief explanation"
}

Rules:
- icd10_code must have NO dots, all uppercase (e.g. "G7001")
- icd10_code_dotted must have the standard dot (e.g. "G70.01")
- Pick the most specific matching code available
- If no exact match exists, pick the closest reasonable code and lower confidence
- confidence is a float 0.0–1.0
- Return JSON only, no markdown fences, no additional text"""


def _build_user_msg(diagnosis_text: str, gold_reference: str,
                    icd_dict: dict, word_index: dict) -> str:
    lines = [f"Diagnosis text: {diagnosis_text}"]
    if gold_reference:
        lines.append(f"Gold reference (context only, not the answer): {gold_reference}")
    hints = get_icd_hints(diagnosis_text, icd_dict, word_index, n=8)
    if hints:
        lines.append("\nCandidate ICD-10-CM codes from local dictionary (for reference):")
        lines.extend(f"  {h}" for h in hints)
    return "\n".join(lines)


def _parse_llm_json(raw: str) -> Optional[dict]:
    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def map_diagnosis_to_icd(
    diagnosis_text: str,
    gold_reference: str,
    icd_dict: dict,
    word_index: dict,
    client,
    model: str,
    cache: dict,
    api_sleep: float,
) -> dict:
    """Return ICD mapping dict for a diagnosis_text, caching by lowercased text."""
    cache_key = diagnosis_text.strip().lower()
    if cache_key in cache:
        return cache[cache_key]

    user_msg = _build_user_msg(diagnosis_text, gold_reference, icd_dict, word_index)
    result: dict
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _parse_llm_json(raw)
        if parsed and "icd10_code" in parsed:
            code = str(parsed["icd10_code"]).upper().replace(".", "").replace(" ", "")
            result = {
                "icd10_code": code,
                "icd10_code_dotted": parsed.get("icd10_code_dotted", ""),
                "label": parsed.get("label", ""),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reason": parsed.get("reason", ""),
            }
        else:
            logging.warning(
                f"Could not parse LLM JSON for: {diagnosis_text!r}\nRaw: {raw[:300]}"
            )
            result = {
                "icd10_code": "UNKNOWN",
                "icd10_code_dotted": "",
                "label": "",
                "confidence": 0.0,
                "reason": "parse_error",
            }
    except Exception as exc:
        logging.error(f"LLM API error for {diagnosis_text!r}: {exc}")
        result = {
            "icd10_code": "UNKNOWN",
            "icd10_code_dotted": "",
            "label": "",
            "confidence": 0.0,
            "reason": f"api_error: {exc}",
        }

    cache[cache_key] = result
    time.sleep(api_sleep)
    return result


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def compute_icd_distribution(icd_counts: dict, icd_labels: dict) -> dict:
    """
    Given {icd_code: count} and {icd_code: label}, return a distribution dict
    with entropy, mode, and per-code entries sorted by descending probability.
    """
    total = sum(icd_counts.values())
    if total == 0:
        return {
            "total_runs": 0,
            "num_distinct_codes": 0,
            "entropy_bits": 0.0,
            "normalized_entropy": 0.0,
            "mode_icd10_code": None,
            "mode_icd10_label": None,
            "mode_prob": 0.0,
            "distribution": [],
        }

    entropy = 0.0
    mode_code, mode_count = None, 0
    dist_entries = []

    for code, count in sorted(icd_counts.items(), key=lambda x: -x[1]):
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)
        if count > mode_count:
            mode_count, mode_code = count, code
        dist_entries.append({
            "icd10_code": code,
            "icd10_label": icd_labels.get(code, ""),
            "count": count,
            "prob": prob,
        })

    n = len(icd_counts)
    norm_entropy = entropy / math.log2(n) if n > 1 else 0.0

    return {
        "total_runs": total,
        "num_distinct_codes": n,
        "entropy_bits": entropy,
        "normalized_entropy": norm_entropy,
        "mode_icd10_code": mode_code,
        "mode_icd10_label": icd_labels.get(mode_code, "") if mode_code else "",
        "mode_prob": mode_count / total,
        "distribution": dist_entries,
    }


# ---------------------------------------------------------------------------
# Divergence metrics
# ---------------------------------------------------------------------------

def _smooth_normalize(prob_dict: dict, support: list, epsilon: float) -> list:
    """Laplace-smooth and renormalize a distribution over a fixed support."""
    vec = [prob_dict.get(k, 0.0) + epsilon for k in support]
    total = sum(vec)
    return [v / total for v in vec]


def _kl(p: list, q: list) -> float:
    """KL(P ‖ Q) in bits. Assumes no zero entries (caller must smooth)."""
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def _js(p: list, q: list) -> float:
    """Jensen-Shannon divergence in [0, 1] (log base 2)."""
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_pairwise_metrics(dist_a: dict, dist_b: dict, epsilon: float) -> dict:
    """
    Compare two ICD distributions. Support = union of codes in both.
    Returns KL(A‖B), KL(B‖A), symmetric KL, JS divergence, JS similarity.
    """
    pa = {e["icd10_code"]: e["prob"] for e in dist_a.get("distribution", [])}
    pb = {e["icd10_code"]: e["prob"] for e in dist_b.get("distribution", [])}
    support = sorted(set(pa) | set(pb))

    if not support:
        return {
            "support_size": 0,
            "kl_ab": 0.0, "kl_ba": 0.0, "sym_kl": 0.0,
            "js_divergence": 0.0, "js_similarity": 1.0,
        }

    p = _smooth_normalize(pa, support, epsilon)
    q = _smooth_normalize(pb, support, epsilon)
    kl_ab = _kl(p, q)
    kl_ba = _kl(q, p)
    sym_kl = (kl_ab + kl_ba) / 2
    js = _js(p, q)

    return {
        "support_size": len(support),
        "kl_ab": kl_ab,
        "kl_ba": kl_ba,
        "sym_kl": sym_kl,
        "js_divergence": js,
        "js_similarity": 1.0 - js,
    }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write_json(path: str, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_csv(path: str, rows: list, fieldnames: Optional[list] = None):
    if not rows:
        logging.warning(f"No rows to write for {path}")
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Categorize mode
# ---------------------------------------------------------------------------

def categorize_folder(
    folder: str,
    case_ids: list,
    run_ids: list,
    temperature: str,
    args,
    icd_dict: dict,
    word_index: dict,
    cache: dict,
) -> dict:
    """
    For each selected case and run in a results folder, call LLM to map
    diagnosis_text → ICD-10-CM, then recompute the ICD-level distribution.

    Returns {case_id: icd_case_data_dict}.
    """
    results_root = Path(args.results_dir)
    out_root = (
        Path(args.out_dir) / args.run_name / "categorized" / folder / temperature
    )
    folder_results: dict = {}
    client = None  # lazy init — only created if an LLM call is needed

    for case_id in case_ids:
        case_file = results_root / folder / temperature / f"case_{case_id}.json"
        out_file = out_root / f"case_{case_id}.icd10.json"

        # ---- Reuse existing output if requested ----
        if args.reuse_existing_categorization and out_file.exists():
            logging.info(f"[{folder}] case_{case_id}: reusing {out_file}")
            try:
                with open(out_file, encoding="utf-8") as f:
                    folder_results[case_id] = json.load(f)
            except json.JSONDecodeError as e:
                logging.warning(f"[{folder}] cached {out_file.name} is malformed ({e}), re-categorizing")
            else:
                continue

        if not case_file.exists():
            logging.warning(f"[{folder}] case_{case_id}.json not found, skipping")
            continue

        try:
            with open(case_file, encoding="utf-8") as f:
                case_data = json.load(f)
        except json.JSONDecodeError as e:
            logging.warning(f"[{folder}] case_{case_id}.json is malformed or empty ({e}), skipping")
            continue

        gold_ref = case_data.get("correct_diagnosis_reference", "")
        run_by_id = {s["run"]: s for s in case_data.get("samples", [])}

        icd_counts: dict = defaultdict(int)
        icd_labels: dict = {}
        mapped_samples = []

        for run_id in run_ids:
            if run_id not in run_by_id:
                logging.warning(
                    f"[{folder}] case_{case_id} run {run_id} not found, skipping"
                )
                continue

            diag_text = run_by_id[run_id].get("diagnosis_text", "")

            # Lazy LLM client creation
            if client is None:
                client = make_llm_client(args.api_key)

            mapping = map_diagnosis_to_icd(
                diag_text, gold_ref, icd_dict, word_index,
                client, args.llm_model, cache, args.api_sleep,
            )

            code = mapping["icd10_code"]
            icd_counts[code] += 1
            if code not in icd_labels:
                # Prefer local dictionary label over LLM-generated label
                icd_labels[code] = icd_dict.get(code, mapping.get("label", ""))

            mapped_samples.append({
                "run": run_id,
                "diagnosis_text": diag_text,
                "icd10_code": code,
                "icd10_code_dotted": mapping.get("icd10_code_dotted", ""),
                "icd10_label": icd_labels[code],
                "confidence": mapping.get("confidence", 0.0),
                "reason": mapping.get("reason", ""),
            })

        dist = compute_icd_distribution(dict(icd_counts), icd_labels)
        icd_case = {
            "scenario_id": case_id,
            "correct_diagnosis_reference": gold_ref,
            "folder": folder,
            "temperature": temperature,
            "selected_runs": run_ids,
            **dist,
            "samples": mapped_samples,
        }

        _write_json(str(out_file), icd_case)
        folder_results[case_id] = icd_case
        logging.info(
            f"[{folder}] case_{case_id}: {dist['num_distinct_codes']} ICD codes | "
            f"mode={dist['mode_icd10_code']} ({dist['mode_prob']:.2f}) | "
            f"entropy={dist['entropy_bits']:.3f} bits"
        )

    return folder_results


def write_folder_summary(
    folder: str,
    folder_results: dict,
    temperature: str,
    args,
):
    out_dir = Path(args.out_dir) / args.run_name / "categorized" / folder
    rows = []
    for case_id in sorted(folder_results):
        r = folder_results[case_id]
        rows.append({
            "scenario_id": case_id,
            "correct_diagnosis_reference": r.get("correct_diagnosis_reference", ""),
            "num_distinct_codes": r.get("num_distinct_codes", 0),
            "entropy_bits": round(r.get("entropy_bits", 0.0), 6),
            "normalized_entropy": round(r.get("normalized_entropy", 0.0), 6),
            "mode_icd10_code": r.get("mode_icd10_code", ""),
            "mode_icd10_label": r.get("mode_icd10_label", ""),
            "mode_prob": round(r.get("mode_prob", 0.0), 6),
            "total_runs": r.get("total_runs", 0),
        })

    summary = {
        "folder": folder,
        "temperature": temperature,
        "run_name": args.run_name,
        "cases": rows,
    }
    _write_json(str(out_dir / "summary.icd10.json"), summary)
    _write_csv(str(out_dir / "summary.icd10.csv"), rows)
    logging.info(f"[{folder}] Folder summary written to {out_dir}/")


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def _load_categorized_case(
    folder: str, case_id: int, temperature: str, args
) -> Optional[dict]:
    path = (
        Path(args.out_dir)
        / args.run_name
        / "categorized"
        / folder
        / temperature
        / f"case_{case_id}.icd10.json"
    )
    if not path.exists():
        logging.warning(f"Categorized file missing: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_folders(
    folders: list,
    case_ids: list,
    temperature: str,
    args,
    all_folder_results: dict,
) -> dict:
    """
    Pairwise comparison of all folder combinations across selected cases.

    Returns:
      pairwise_rows   – one row per (folder_a, folder_b, case_id)
      js_matrix       – folder × folder JS similarity (averaged over cases)
      sym_kl_matrix   – folder × folder symmetric KL (averaged over cases)
    """
    pairwise_rows = []
    # Accumulate per-pair values across cases
    pair_js: dict = defaultdict(list)
    pair_sym_kl: dict = defaultdict(list)

    for i, fa in enumerate(folders):
        for j, fb in enumerate(folders):
            if j <= i:
                continue
            for case_id in case_ids:
                da = all_folder_results.get(fa, {}).get(case_id)
                db = all_folder_results.get(fb, {}).get(case_id)
                if da is None or db is None:
                    logging.warning(
                        f"Skipping pair ({fa}, {fb}) case {case_id}: missing data"
                    )
                    continue
                m = compute_pairwise_metrics(da, db, args.epsilon)
                pairwise_rows.append({
                    "folder_a": fa,
                    "folder_b": fb,
                    "case_id": case_id,
                    "support_size": m["support_size"],
                    "kl_ab": round(m["kl_ab"], 6),
                    "kl_ba": round(m["kl_ba"], 6),
                    "sym_kl": round(m["sym_kl"], 6),
                    "js_divergence": round(m["js_divergence"], 6),
                    "js_similarity": round(m["js_similarity"], 6),
                })
                pair_js[(fa, fb)].append(m["js_divergence"])
                pair_sym_kl[(fa, fb)].append(m["sym_kl"])

    # Build full matrices (symmetric for JS similarity / sym KL)
    js_matrix: dict = {}
    sym_kl_matrix: dict = {}
    for fa in folders:
        js_matrix[fa] = {}
        sym_kl_matrix[fa] = {}
        for fb in folders:
            if fa == fb:
                js_matrix[fa][fb] = 1.0
                sym_kl_matrix[fa][fb] = 0.0
            else:
                # Look up in both orderings (we only stored (fa,fb) with i<j)
                vals_js = pair_js.get((fa, fb)) or pair_js.get((fb, fa)) or []
                vals_kl = pair_sym_kl.get((fa, fb)) or pair_sym_kl.get((fb, fa)) or []
                js_sim = round(1.0 - sum(vals_js) / len(vals_js), 6) if vals_js else None
                avg_kl = round(sum(vals_kl) / len(vals_kl), 6) if vals_kl else None
                js_matrix[fa][fb] = js_sim
                sym_kl_matrix[fa][fb] = avg_kl

    return {
        "pairwise_rows": pairwise_rows,
        "js_matrix": js_matrix,
        "sym_kl_matrix": sym_kl_matrix,
    }


def write_compare_outputs(compare_data: dict, folders: list, args):
    out_dir = Path(args.out_dir) / args.run_name / "compare"

    # Per-case pairwise metrics
    _write_csv(
        str(out_dir / "pairwise_case_metrics.csv"),
        compare_data["pairwise_rows"],
        fieldnames=[
            "folder_a", "folder_b", "case_id", "support_size",
            "kl_ab", "kl_ba", "sym_kl", "js_divergence", "js_similarity",
        ],
    )

    # JS similarity matrix
    js_mat = compare_data["js_matrix"]
    js_rows = [{"folder": fa, **{fb: js_mat[fa][fb] for fb in folders}} for fa in folders]
    _write_csv(
        str(out_dir / "folder_similarity_matrix_js.csv"),
        js_rows,
        fieldnames=["folder"] + folders,
    )

    # Symmetric KL matrix
    kl_mat = compare_data["sym_kl_matrix"]
    kl_rows = [{"folder": fa, **{fb: kl_mat[fa][fb] for fb in folders}} for fa in folders]
    _write_csv(
        str(out_dir / "folder_similarity_matrix_symmetric_kl.csv"),
        kl_rows,
        fieldnames=["folder"] + folders,
    )

    # Combined JSON
    _write_json(
        str(out_dir / "folder_similarity_matrix.json"),
        {
            "folders": folders,
            "run_name": args.run_name,
            "js_similarity_matrix": js_mat,
            "symmetric_kl_matrix": kl_mat,
        },
    )
    logging.info(f"Compare outputs written to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()

    case_ids = parse_id_spec(args.cases)
    run_ids = parse_id_spec(args.runs)

    logging.info(f"Mode      : {args.mode}")
    logging.info(f"Folders   : {args.folders}")
    logging.info(f"Cases     : {case_ids[:5]}{'...' if len(case_ids) > 5 else ''} ({len(case_ids)} total)")
    logging.info(f"Runs      : {run_ids}")
    logging.info(f"Temp      : {args.temperature}")
    logging.info(f"Run name  : {args.run_name}")

    icd_dict = load_icd_dict(args.icd_dict)
    word_index = build_word_index(icd_dict) if icd_dict else {}
    cache = load_cache(args.cache_file)

    all_folder_results: dict = {}

    # ------------------------------------------------------------------ #
    # CATEGORIZE                                                           #
    # ------------------------------------------------------------------ #
    if args.mode in ("categorize", "both"):
        for folder in args.folders:
            logging.info(f"=== Categorizing: {folder} ===")
            results = categorize_folder(
                folder, case_ids, run_ids, args.temperature,
                args, icd_dict, word_index, cache,
            )
            all_folder_results[folder] = results
            write_folder_summary(folder, results, args.temperature, args)
            # Persist cache after every folder to guard against interruption
            save_cache(cache, args.cache_file)
            logging.info(
                f"[{folder}] Categorized {len(results)}/{len(case_ids)} cases"
            )

    # ------------------------------------------------------------------ #
    # COMPARE                                                              #
    # ------------------------------------------------------------------ #
    if args.mode in ("compare", "both"):
        if len(args.folders) < 2:
            logging.error(
                "Compare mode requires at least 2 --folders. "
                "Got: %s", args.folders
            )
            sys.exit(1)

        # Load any folders not already in memory (compare-only mode)
        for folder in args.folders:
            if folder in all_folder_results:
                continue
            logging.info(f"Loading categorized data for: {folder}")
            folder_data: dict = {}
            for case_id in case_ids:
                data = _load_categorized_case(folder, case_id, args.temperature, args)
                if data is not None:
                    folder_data[case_id] = data
            all_folder_results[folder] = folder_data
            logging.info(
                f"[{folder}] Loaded {len(folder_data)}/{len(case_ids)} cases"
            )

        logging.info("=== Running pairwise comparison ===")
        compare_data = compare_folders(
            args.folders, case_ids, args.temperature, args, all_folder_results
        )
        write_compare_outputs(compare_data, args.folders, args)

    logging.info("Done.")


if __name__ == "__main__":
    main()
