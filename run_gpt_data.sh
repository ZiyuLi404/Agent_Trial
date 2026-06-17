#!/usr/bin/env bash
# Generate GPT-family diagnosis-distribution data for the embedding SNR experiment.
#
# Concurrency model (per user request): ONE OS PROCESS PER CASE.
#   - Each case is its own `diagnosis_distribution.py` process with --concurrency 1,
#     so the 10 repeats inside a case run strictly serially (no in-process thread
#     pool) -> parallelism never touches a single case's results.
#   - Parallelism is purely across cases (9 procs per batch) and across models.
#   - Each per-case process writes to its OWN out_dir so the per-run summary.json
#     writes can't race.
#
# Layout of parallelism:
#   - All 4 batches (2 models x 2 batches) run in parallel.
#   - Within a batch, the 9 cases run in parallel.
#   - => up to 4 x 9 = 36 case processes at once. gpt-5.4-mini's 18 concurrent
#     streams may briefly exceed its 200K TPM; the engine's retry/backoff absorbs
#     the 429s (correctness unaffected, just possibly slower).
#
# Output is assembled into Data_gpt/ matching the existing Data.zip layout.
set -uo pipefail

cd "$(dirname "$0")"
set -a; source .env; set +a

PY=.venv/bin/python
CASES=(2 5 15 18 23 24 25 26 28)
RUNS=10
PATIENT="deepseek-v4-flash"

# Run one batch: 9 case processes in parallel, wait for all.
run_batch() {
  local model="$1" tag="$2"
  echo "================ START $tag (doctor=$model) $(date) ================"
  mkdir -p "results/gpt/$tag"
  local pids=() c
  for c in "${CASES[@]}"; do
    $PY -u diagnosis_distribution.py \
      --dataset MedQA --scenario_ids "$c" --runs "$RUNS" \
      --doctor_llm "$model" --patient_llm "$PATIENT" --measurement_llm "$model" \
      --bucketing exact --temperatures 0.05 --total_inferences 20 \
      --concurrency 1 --out_dir "results/gpt/$tag/case$c" \
      > "results/gpt/$tag/case$c.log" 2>&1 &
    pids+=($!)
  done
  local fail=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=1; fi
  done
  [ "$fail" -ne 0 ] && echo "!!!!!!!! $tag had a failed case; check results/gpt/$tag/*.log"
  echo "================ DONE  $tag $(date) ================"
}

# Launch all 4 batches in parallel.
run_batch "gpt-5.5"      "gpt_5_5_1"      &
run_batch "gpt-5.5"      "gpt_5_5_2"      &
run_batch "gpt-5.4-mini" "gpt_5_4_mini_1" &
run_batch "gpt-5.4-mini" "gpt_5_4_mini_2" &
wait

# Assemble Data_gpt/<tag>/case_<id>.json from each per-case out_dir.
echo "================ ASSEMBLING Data_gpt/ ================"
rm -rf Data_gpt
TAGS=(gpt_5_5_1 gpt_5_5_2 gpt_5_4_mini_1 gpt_5_4_mini_2)
missing=0
for tag in "${TAGS[@]}"; do
  mkdir -p "Data_gpt/$tag"
  for c in "${CASES[@]}"; do
    src="results/gpt/$tag/case$c/temp_0.05/case_$c.json"
    if [ -f "$src" ]; then
      cp "$src" "Data_gpt/$tag/case_$c.json"
    else
      echo "MISSING: $src"; missing=1
    fi
  done
done

echo "================ ZIPPING ================"
rm -f Data_gpt.zip
zip -r -q Data_gpt.zip Data_gpt
[ "$missing" -ne 0 ] && echo "WARNING: some case files were missing (see MISSING lines above)"
echo "ALL_DONE: $(date)"
ls -R Data_gpt
