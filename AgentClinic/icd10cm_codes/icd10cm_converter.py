import json
from pathlib import Path

input_path = Path("icd10cm_codes/icd10cm-codes-April-1-2026.txt")
output_path = Path("icd10cm_codes/icd10cm_2026.jsonl")

with input_path.open("r", encoding="utf-8", errors="ignore") as f, \
     output_path.open("w", encoding="utf-8") as out:

    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue

        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue

        code, description = parts
        record = {
            "code": code.strip(),
            "description": description.strip()
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")