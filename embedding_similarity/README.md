# embedding_similarity — Diagnosis Drift, Method B (vectors + similarity) (G)

**Detects diagnosis drift between models/versions by embedding the text into vectors and comparing them with cosine similarity. Parallel to `kl_js_divergence` (Method A, distributions + divergence).**

Offline analysis — no consultations, no engine (just sentence-transformers / numpy / pandas / sklearn). Input is the `case_X.json` produced by **`generate_diagnosis_distribution` (F)**.

## Two parallel experiments — choose with `--text_field` (required, no default)
| `--text_field` | Embeds | Answers |
|----------------|--------|---------|
| `diagnosis_text` | the final diagnosis line | do two models reach the same **conclusion**? |
| `full_dialogue` | the whole consultation | do two models follow the same **process**? |

## Run
```bash
# from the repo root; data dir holds <group>/case_*.json
python embedding_similarity/embedding_similarity.py \
    --data_dir results/generate_diagnosis_distribution \
    --text_field diagnosis_text \
    --model Qwen/Qwen3-Embedding-0.6B
# output defaults to results/embedding_similarity/<text_field>/
```
Model groups are **inferred from folder names** by stripping the trailing batch number `_<run>`: `deepseek_flash_1/2 → deepseek_flash`, `gpt_5_5_1/2 → gpt_5_5`. **Adding a new model needs no code change.**

## Output
`mean_group_similarity_matrix.csv`, `mean_model_similarity_matrix.csv`, `case_level_*`, `*.npz`, etc. The similarity matrices feed **`history_borrowing` (H)**.

## Refactor notes
- ✅ **Three scripts merged into one** `embedding_similarity.py`: the only real difference (which text to embed) became `--text_field`.
- ✅ **Reads a directory** (`--data_dir`) instead of `Data*.zip`; the `zipfile` + `__MACOSX/._` filtering is gone.
- ✅ **No hardcoded tables**: dropped `MODEL_ORDER` / keyword-based detection; groups are inferred from folder names.
- ✅ **Verified**: the new loader matches the old ones byte-for-byte on `(group, case, run, text)` for both `full_dialogue` (900 rows) and `diagnosis_text` (gpt, 360 rows); the similarity math is copied verbatim. Model-level labels are now more descriptive (`deepseek_flash` vs old `flash`); grouping and values are unchanged.
- Old `Data*.zip` archived under `generated_outputs/`; unzipped data lives in `results/generate_diagnosis_distribution/`. Old result dirs moved to `results/embedding_similarity/`.
