#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild the 5-model consultation dataset for lora_fingerprint.

The shipped Data.zip only contains 3 of the 5 model families. The full 5-model
full_dialogue text (incl. gpt_5_5 / gpt_5_4_mini) survives in the embedding CSV:
    results/embedding_similarity/embedding_full_dialogue_results_gpt_8b/raw_full_dialogue_outputs.csv

This script turns that CSV back into the per-group case_<id>.json layout that
fingerprint_detector.py expects:
    results/generate_diagnosis_distribution/<group>/case_<id>.json
where each json has {"samples": [{"full_dialogue": ..., "diagnosis_text": ...}, ...]}.

Usage (from repo root):
    python lora_fingerprint/reconstruct_5model_data.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SRC = Path("results/embedding_similarity/embedding_full_dialogue_results_gpt_8b/raw_full_dialogue_outputs.csv")
OUT = Path("results/generate_diagnosis_distribution")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source CSV not found: {SRC}")

    data: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    ref: dict[tuple[str, str], str] = {}
    with open(SRC, newline="") as f:
        for row in csv.DictReader(f):
            grp, cid = row["group"], row["case_id"]
            data[grp][cid].append({
                "run": row.get("run_id"),
                "diagnosis_text": row.get("diagnosis_text_for_reference_only", ""),
                "full_dialogue": row.get("full_dialogue", ""),
            })
            ref[(grp, cid)] = row.get("correct_diagnosis_reference", "")

    written = 0
    for grp, cases in data.items():
        d = OUT / grp
        d.mkdir(parents=True, exist_ok=True)
        for cid, samples in cases.items():
            samples.sort(key=lambda s: int(s["run"]) if str(s.get("run")).isdigit() else 0)
            obj = {
                "scenario_id": int(cid) if str(cid).isdigit() else cid,
                "correct_diagnosis_reference": ref[(grp, cid)],
                "samples": samples,
            }
            (d / f"case_{cid}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            written += 1
    print(f"wrote {written} case files across {len(data)} groups -> {OUT}")
    for grp in sorted(data):
        print(f"  {grp}: {len(data[grp])} cases")


if __name__ == "__main__":
    main()
