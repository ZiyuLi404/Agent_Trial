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

## Scaling up to Qwen2.5-7B + LoRA (cloud GPU)

The same script drives a decoder-LM backbone (Qwen2.5 / Llama), matching the FDLLM paper. It is **not** for the 16GB Mac — run it on a CUDA GPU (Colab / A100 / lab box). The script auto-handles the decoder-LM bits (pad token, `q_proj,v_proj` LoRA targets, 4-bit QLoRA, gradient checkpointing).

```bash
# on the GPU box, from repo root:
pip install -r lora_fingerprint/requirements.txt   # adds peft / accelerate / bitsandbytes
bash lora_fingerprint/run_qwen_cloud.sh            # Qwen2.5-7B + QLoRA, scenario split
```

Equivalent explicit call:
```bash
python lora_fingerprint/fingerprint_detector.py \
  --model_name Qwen/Qwen2.5-7B --allow_remote_model_files \
  --text_field full_dialogue --split_mode scenario \
  --use_lora --load_in_4bit --gradient_checkpointing --dtype bfloat16 \
  --lora_r 256 --lora_alpha 128 --learning_rate 1e-4 --epochs 3 \
  --batch_size 2 --max_length 1024 \
  --output_dir results/lora_fingerprint/qwen7b_scenario
```
`--load_in_4bit` (QLoRA, needs `bitsandbytes` + CUDA) fits 7B on a single 16GB card; drop it on 24GB+ and raise `--batch_size` / `--max_length`. `--dtype bfloat16` is the right choice for Qwen/Llama LoRA on a GPU.

> ⚠️ Expectation: on the **current** 5-model, fixed-prompt, temp-0.05 data the task is already saturated (TF-IDF bag-of-words ≈ DistilBERT ≈ LoRA ≈ 0.91), so a 7B backbone is unlikely to beat ~0.91 here. Qwen earns its keep when the task gets harder — more model classes, Chinese/multilingual, higher temperature, or adversarial paraphrase/translation (the regime FDLLM targets).

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

## Pairwise comparisons across many model versions

`fingerprint_detector.py` trains one classifier over *all* labels found under `--data_dir` (an N-way problem). For "how separable is version A from version B specifically" — e.g. comparing 5 model versions two at a time — use the two files below instead. Neither duplicates the training loop; they import the reusable pieces (`load_examples`, `split_examples`, `load_backbone`, `train_model`, …) straight from `fingerprint_detector.py`.

- **`pairwise_fingerprint.py`** — trains one binary `version_a` vs `version_b` classifier and writes `metadata.json` / `pairwise_outputs.jsonl` (per-dialogue logits, probabilities, confidence, margin, entropy) / `summary_metrics.json` (accuracy, macro-F1, AUROC, …) under `<output_dir>/<version_a>__vs__<version_b>/`.
- **`analyze_pairwise_fingerprint.py`** — runs `pairwise_fingerprint.run_pairwise_fingerprint` over every pair among a set of versions (or an explicit list of pairs) and aggregates the results into analysis-wide CSVs.

### Run all pairwise comparisons for 5 model versions

```bash
python lora_fingerprint/analyze_pairwise_fingerprint.py \
  --data_dir results/generate_diagnosis_distribution \
  --output_dir results/lora_fingerprint_pairwise \
  --analysis_dir results/lora_fingerprint_pairwise_analysis \
  --versions deepseek_flash,deepseek_pro,gpt_5_4_mini,gpt_5_5,qwen_plus \
  --text_field full_dialogue --split_mode scenario \
  --skip_existing
```

5 versions → `itertools.combinations` → all 10 unordered pairs (`deepseek_flash vs deepseek_pro`, `deepseek_flash vs gpt_5_4_mini`, … `gpt_5_5 vs qwen_plus`). `--skip_existing` avoids retraining a pair whose 3 output files are already on disk (fast — no model load); `--overwrite` forces a fresh run of every pair regardless.

To run only specific pairs instead of the full combinatorial set:
```bash
python lora_fingerprint/analyze_pairwise_fingerprint.py \
  --data_dir results/generate_diagnosis_distribution \
  --output_dir results/lora_fingerprint_pairwise \
  --analysis_dir results/lora_fingerprint_pairwise_analysis \
  --pairs deepseek_flash:deepseek_pro,gpt_5_4_mini:gpt_5_5 \
  --text_field full_dialogue --split_mode scenario
```

To also compute per-dialogue "borrowing" features relative to one target version (e.g. how distinguishable each other version is from `gpt_5_5`, at several exponential-decay strengths):
```bash
python lora_fingerprint/analyze_pairwise_fingerprint.py \
  --data_dir results/generate_diagnosis_distribution \
  --output_dir results/lora_fingerprint_pairwise \
  --analysis_dir results/lora_fingerprint_pairwise_analysis_gpt55 \
  --versions deepseek_flash,deepseek_pro,gpt_5_4_mini,gpt_5_5,qwen_plus \
  --target_version gpt_5_5 --lambda_values 1,5,10,50,100 \
  --text_field full_dialogue --split_mode scenario
```

All training pass-through flags from `fingerprint_detector.py` (`--model_name`, `--use_lora`, `--load_in_4bit`, `--max_length`, `--epochs`, `--batch_size`, `--learning_rate`, `--weight_decay`, `--warmup_ratio`, `--seed`) are supported. One failing pair is logged and skipped, not fatal — pass `--fail_fast` to stop the whole run on the first failure instead. A completed/skipped/failed summary prints at the end.

### Outputs

Per pair, under `<output_dir>/<version_a>__vs__<version_b>/`:
- `metadata.json` — split spec, model/training config, label map (setup only, no scores)
- `pairwise_outputs.jsonl` — one row per dialogue (train + test) with logits, `p_a`/`p_b`, confidence, margin, entropy, diagnosis text
- `summary_metrics.json` — accuracy / macro-F1 / balanced accuracy / AUROC, overall + by split + by true version + by case

Aggregated across all pairs, under `<analysis_dir>/`:
- `all_pair_summary.csv` / `.json` — one row per pair
- `all_dialog_features.csv` — every pair's `pairwise_outputs.jsonl` rows concatenated
- `borrowing_features.csv` — raw candidate features (`fingerprint_distance_prob`, `exp_weight_prob_lambda_<L>`, …) for a downstream borrowing-weight decision — this script does not pick a final weight, it only produces the inputs
- `analysis_metadata.json` — which pairs were requested/completed/skipped/failed

### Run just one pair (skip the multi-pair runner)

```bash
python lora_fingerprint/pairwise_fingerprint.py \
  --data_dir results/generate_diagnosis_distribution \
  --version_a deepseek_flash --version_b deepseek_pro \
  --output_dir results/lora_fingerprint_pairwise \
  --text_field full_dialogue --split_mode scenario
```
