# kl_js_divergence — Diagnosis Drift, Method A (distributions + divergence) (I)

**Detects diagnosis drift between models/prompts by mapping free-text diagnoses to standard ICD-10-CM codes, then comparing the code distributions with KL / JS divergence. Parallel to `embedding_similarity` (Method B, vectors + similarity).**

Offline analysis — no consultations, no engine (it has its own LLM client). A **terminal branch**: its output goes straight to the researcher (and figures), not to any downstream module.

## Two steps
1. **Categorize** — an LLM maps each diagnosis (`"pneumonia"`, `"lung infection"`, `"肺炎"`, …) to a standard ICD-10-CM code, so different wordings of the same disease line up.
2. **Compare** — recompute the distribution at the ICD level, then compare folders pairwise with **KL / JS divergence**.

## Output
- `summary.icd10.csv` — each folder's distribution over ICD codes
- `folder_similarity_matrix_js.csv`, `folder_similarity_matrix_symmetric_kl.csv` — pairwise divergence
- `pairwise_case_metrics.csv` — per-case metrics

## Three ready-made comparisons (see `results/kl_js_divergence/`)
| Folder | Compares |
|--------|----------|
| `deepseek_flash_vs_pro/` | same family, different tiers |
| `deepseek_vs_gpt/` | across vendors |
| `prompt_compare/` | different prompt personas (the **prompt experiment**) |

## Run
```bash
# from the repo root
python kl_js_divergence/icd_categorize_compare.py \
    --mode both --folders 50case_10runs_flash \
    --cases 0-19 --runs 0-9 --temperature temp_0.05 --run_name analysis_v1
```

## Dependencies
- ICD dictionary: uses the single source `AgentClinic/icd10cm_2026.jsonl` (deduped — no local copy here).
- LLM: needs `OPENAI_API_KEY` (to map diagnoses to ICD codes).

## Refactor notes
- ✅ Renamed `result_categorize → kl_js_divergence`; script `icd10_categorize_compare.py → icd_categorize_compare.py`.
- ✅ ICD dictionary deduped; `--icd_dict` defaults to `AgentClinic/icd10cm_2026.jsonl`.
- ✅ Comparison data moved to `results/kl_js_divergence/`; `--out_dir` / `--cache_file` default there. This folder now holds only code + README.
