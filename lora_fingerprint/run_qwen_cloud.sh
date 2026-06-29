#!/usr/bin/env bash
# Qwen2.5-7B + LoRA fingerprint detector — run on a CUDA GPU (Colab / A100 / lab box).
# NOT for the 16GB Mac: a 7B base needs a real GPU. With QLoRA (--load_in_4bit)
# it fits on a single 16GB card (T4/V100); on a 24GB+ card you can drop 4-bit.
#
# Setup (once, on the GPU box):
#   pip install -r lora_fingerprint/requirements.txt        # peft, accelerate, bitsandbytes...
#   # make sure results/generate_diagnosis_distribution/ is present (unzip generated_outputs/Data*.zip)
#
# Run from the repo root:  bash lora_fingerprint/run_qwen_cloud.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"          # base model FDLLM uses; -Instruct also works
OUT="${OUT:-results/lora_fingerprint/qwen7b_scenario}"

python lora_fingerprint/fingerprint_detector.py \
  --model_name "$MODEL" \
  --allow_remote_model_files \
  --text_field full_dialogue \
  --split_mode scenario --test_size 0.33 --seed 42 \
  --use_lora --load_in_4bit --gradient_checkpointing --dtype bfloat16 \
  --lora_r 256 --lora_alpha 128 --lora_dropout 0.1 \
  --learning_rate 1e-4 --epochs 3 --batch_size 2 --max_length 1024 \
  --logging_steps 20 \
  --output_dir "$OUT"

# Notes:
# - Hyperparameters above mirror the FDLLM paper (r=256, alpha=128, epoch=3, bs=2, lr=1e-4).
#   For a quicker/cheaper run use --lora_r 16 --lora_alpha 32 --lora_dropout 0.05.
# - On a 24GB+ GPU you can drop --load_in_4bit and raise --batch_size / --max_length 2048.
# - split_mode=scenario gives the honest held-out-case number; use --split_mode batch
#   only to compare directly against the old same-scenario runs.
