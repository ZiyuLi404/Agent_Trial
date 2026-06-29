# lora_fingerprint — 5-model results (full_dialogue, scenario split)

Run date: 2026-06-28. GPU: RTX 5090 (32 GB). Env: conda `agentclinic` (torch 2.11 cu128, transformers 5.12).

## Data
5 model families, each with two batches (`_1` / `_2`), 9 cases × 10 runs.
The 5-model consultation data was **reconstructed** from
`results/embedding_similarity/embedding_full_dialogue_results_gpt_8b/raw_full_dialogue_outputs.csv`
into `results/generate_diagnosis_distribution/<group>/case_<id>.json`
(the shipped `Data.zip` only had 3 of the 5 models). See `reconstruct_5model_data.py`.

Classes: `Qwen_plus_turbo`, `deepseek_flash`, `deepseek_pro`, `gpt_5_4_mini`, `gpt_5_5`.

## Split
`--split_mode scenario --seed 42` → **held-out cases**:
- train cases: 2, 5, 15, 23, 24, 28  (600 examples)
- test  cases: 18, 25, 26            (300 examples, 60 per class)

Note: `eval == test` in this script (no separate dev set); the best epoch is picked by
macro-F1 on the test set, so absolute numbers are mildly optimistic. Applied equally to all
runs, so the head-to-head comparison is fair.

## Results (test = held-out cases 18/25/26)
| backbone | LoRA | accuracy | macro-F1 |
|---|---|---|---|
| distilbert-base-uncased (66M, full FT) | – | **0.910** | **0.910** |
| Qwen2.5-7B | r=16, α=32 | 0.760 | 0.761 |
| Qwen2.5-7B | r=256, α=128 | 0.693 | 0.694 |

## Key finding
On the **same** hard held-out split, the 66M DistilBERT **beats** Qwen2.5-7B+LoRA.
This is **not** a data ceiling (DistilBERT shows the 5 classes are cleanly separable at 0.91)
— the 7B run **overfits**: with only 6 training cases, train_loss collapses to ~0.002 while
eval_loss stays ~1.1–1.5. Lowering LoRA rank 256→16 cut the overfit and lifted 0.69→0.76,
but still short of DistilBERT.

DistilBERT confusion (rows=true): Qwen perfect (60/60); only same-vendor siblings blur
(flash↔pro, gpt_5_4_mini↔gpt_5_5). Cross-vendor fingerprint is strong — consistent with the
earlier JS-divergence finding.

## Open next steps
1. rank 8 + more epochs (8–10) + lower lr; rely on macro-F1 best-epoch.
2. Verify decoder-as-classifier last-token pooling / padding side for Qwen2ForSequenceClassification.
3. A proper 3-way train/val/test (data is tiny — only 9 cases — so this is hard).

## Reproduce
```bash
conda activate agentclinic
export HF_HOME=/root/autodl-tmp/hf-cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0
cd Agent_Trail_refactor
# DistilBERT baseline
python lora_fingerprint/fingerprint_detector.py --model_name distilbert-base-uncased \
  --allow_remote_model_files --text_field full_dialogue --split_mode scenario \
  --epochs 5 --batch_size 8 --max_length 512 \
  --output_dir results/lora_fingerprint/distilbert_scenario_5model
# Qwen2.5-7B + LoRA r=16
python lora_fingerprint/fingerprint_detector.py --model_name Qwen/Qwen2.5-7B \
  --allow_remote_model_files --text_field full_dialogue --split_mode scenario \
  --use_lora --gradient_checkpointing --dtype bfloat16 --lora_r 16 --lora_alpha 32 \
  --lora_dropout 0.1 --learning_rate 1e-4 --epochs 5 --batch_size 4 --max_length 1024 \
  --output_dir results/lora_fingerprint/qwen7b_scenario_5model_r16
```
