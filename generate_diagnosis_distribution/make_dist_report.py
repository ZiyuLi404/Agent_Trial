#!/usr/bin/env python3
"""Generate English diagnosis-distribution reports for each result group."""
import json
import os
import glob

# results/ lives at the repo root, not inside this module dir.
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)

GROUPS = [
    "diag_dist_flash_0-39",
    "diag_dist_pro_0-19",
    "diag_dist_pro_40-49",
]


def format_case(data, src_path):
    dist = data.get("distribution", [])
    N = sum(item["count"] for item in dist)
    distinct = data.get("num_distinct_buckets", len(dist))
    entropy = data.get("entropy_bits", 0.0)

    lines = []
    scen = data.get("scenario_id")
    correct = data.get("correct_diagnosis_reference", "")
    lines.append(f"===== case_{scen} | correct: {correct} =====")
    lines.append(
        f"Diagnosis distribution (N={N}, distinct={distinct}, entropy={entropy:.3f} bits):"
    )
    for item in sorted(dist, key=lambda x: x["count"], reverse=True):
        count = item["count"]
        pct = 100.0 * count / N if N else 0.0
        text = item.get("canonical", "")
        pct_str = f"{pct:.1f}%"
        lines.append(f"\u2502{pct_str:>10}  ({count:>2}/{N})  {text}")
    rel = os.path.relpath(src_path, os.path.dirname(RESULTS_DIR))
    lines.append(f"\u2514\u2500 source \u2192 {rel}")
    return "\n".join(lines)


def main():
    for group in GROUPS:
        temp_dir = os.path.join(RESULTS_DIR, group, "temp_0.05")
        if not os.path.isdir(temp_dir):
            print(f"[skip] {group}: no temp_0.05 dir")
            continue
        files = glob.glob(os.path.join(temp_dir, "case_*.json"))
        files.sort(key=lambda p: int(os.path.basename(p)[5:-5]))

        blocks = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            blocks.append(format_case(data, f))

        report = "\n\n".join(blocks) + "\n"
        out_path = os.path.join(RESULTS_DIR, group, "distribution_report.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"### Group: {group} ({len(files)} cases) ###\n\n")
            fh.write(report)
        print(f"[ok] {group}: {len(files)} cases -> {out_path}")


if __name__ == "__main__":
    main()
