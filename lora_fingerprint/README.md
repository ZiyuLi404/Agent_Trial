# lora_fingerprint — Model Fingerprint Detection via LoRA (I)

**Given one consultation text, predict which LLM produced it** (`gpt_5_5` / `deepseek_pro` / `Qwen_plus_turbo` …). A lightweight, in-repo adaptation of **FDLLM** ([arXiv:2501.16029](https://arxiv.org/abs/2501.16029)): instead of comparing models pairwise (`embedding_similarity`, Method B) or comparing distributions (`kl_js_divergence`, Method A), here we **train a text classifier (optionally with LoRA) to learn each model's stylistic fingerprint** and measure how separable the models are.

Offline analysis — no consultations, no engine. Input is the same `case_X.json` produced by **`generate_diagnosis_distribution` (F)**; this module only *reads* that directory and shares nothing else with the other modules.

## Two parallel experiments — choose with `--text_field`
| `--text_field` | Classifies from | Answers |
|----------------|-----------------|---------|
| `full_dialogue` (default) | the whole consultation | is a model's **process** identifiable? |
| `diagnosis_text` | the final diagnosis line | is a model's **conclusion style** identifiable? |

## Train/test split — `--split_mode`
| mode | how | use it for |
|------|-----|-----------|
| `batch` (default) | each family's `_1` folder → train, `_2` → test | honest generalization: train on one batch, test on a held-out batch |
| `random` | label-stratified random split (`--test_size`) | quick sanity check on a single batch |

Labels are **inferred from folder names** by stripping the trailing batch number `_<n>`: `deepseek_flash_1/2 → deepseek_flash`, `gpt_5_5_1/2 → gpt_5_5`. **Adding a new model needs no code change.** (Same convention as `embedding_similarity`.)

## Run
```bash
# from the repo root
# 0) just look at the split / class balance, no training:
python lora_fingerprint/fingerprint_detector.py --prepare_only

# 1) full_dialogue fingerprinting (DistilBERT, full fine-tune):
python lora_fingerprint/fingerprint_detector.py \
    --text_field full_dialogue --epochs 5 --batch_size 4

# 2) LoRA instead of full fine-tune (needs `pip install peft`):
python lora_fingerprint/fingerprint_detector.py \
    --text_field diagnosis_text --use_lora --lora_r 16

# output defaults to results/lora_fingerprint/<text_field>/
```

## Output (under `results/lora_fingerprint/<text_field>/`)
- `metrics.json` — accuracy / macro-F1 + run config
- `classification_report.txt`, `confusion_matrix.csv` — per-model breakdown
- `train_examples.jsonl` / `test_examples.jsonl` — the exact split used
- `best_model/`, `label_map.json` — the saved detector (git-ignored, regenerable)

## Dependencies
`torch`, `transformers`, `scikit-learn`, `numpy` — already in the repo. `--use_lora` additionally needs `peft` (`pip install peft`); without that flag it is not imported. See `requirements.txt`.

## Relationship to the paper (how this differs from FDLLM)
| | FDLLM paper | this module |
|---|---|---|
| backbone | Qwen2.5-7B (generative) | `distilbert-base-uncased` (~66M, English) — runs on a laptop/MPS |
| LoRA | core, `r=256, α=128` | **optional** (`--use_lora`); default is a plain classification head |
| data | FD-Dataset, 90k, bilingual, 5 domains | this repo's medical-consultation dialogues, 5 models |
| classes | 20 LLMs | the model families present under `--data_dir` |

So this is *FDLLM's idea* (LoRA-style fingerprint classification), re-scoped to the Agent-Trial medical-dialogue setting and made dependency-light. The paper's official code is not publicly released (only promised post-publication), so nothing here is copied from it.

## Notes
- DistilBERT is English-only and truncates at `--max_length` (512) tokens; long Chinese dialogues should switch `--model_name` to a multilingual/long-context encoder.
- With only `_1`/`_2` per model, `--split_mode batch` gives a small but honest held-out test; treat absolute numbers as indicative.
