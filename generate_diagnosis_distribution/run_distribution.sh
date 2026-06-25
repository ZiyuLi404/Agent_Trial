#!/usr/bin/env bash
# =============================================================================
# run_distribution.sh — general-purpose runner for diagnosis_distribution.py
# =============================================================================
#
# WHAT IT DOES
#   Runs the diagnosis-distribution generator across any set of doctor models
#   and cases, with one OS process per (model, case) so results never race and
#   a single case's repeats stay reproducible. Parallelism is throttled to a
#   configurable number of concurrent processes (rate-limit friendly).
#
# DESIGN (vs the old run_gpt_data.sh)
#   - Nothing is hardcoded: models / cases / runs / dataset / temps / agents
#     are all configurable via flags or environment variables.
#   - Concurrency is CAPPED (--max-parallel) instead of blindly launching 36
#     processes at once, so you control how hard you hit the API.
#   - Each (model, case) writes to its OWN out_dir -> no write races.
#   - In-case repeats stay serial by default (--in-case-concurrency 1) to keep
#     a single case's distribution reproducible (parallelism only across cases).
#   - Runs from the repo root, uses whatever `python` is active (conda env),
#     and relies on .env (auto-loaded by the Python script) for API keys.
#
# USAGE
#   bash generate_diagnosis_distribution/run_distribution.sh [options]
#
#   Options (all have env-var equivalents in CAPS shown in [brackets]):
#     --models   "a,b"     Doctor LLMs, comma-separated         [MODELS]
#     --cases    "2,5,7-9" Scenario ids: list and/or ranges     [CASES]
#     --runs      N        Repeats per case                     [RUNS]
#     --dataset   NAME     MedQA | MedQA_Ext | NEJM | NEJM_Ext  [DATASET]
#     --temps    "0.05"    Temperatures (passed straight thru)  [TEMPS]
#     --patient   MODEL    Patient LLM                          [PATIENT]
#     --measurement MODEL  Measurement LLM                      [MEASUREMENT]
#     --moderator MODEL    Moderator LLM                        [MODERATOR]
#     --bucketing exact    exact | semantic                     [BUCKETING]
#     --styles   "a,b"     Doctor prompt style(s) or "all"      [STYLES]
#     --prompt-json PATH   Doctor prompt JSON (rel. to AgentClinic/) [PROMPT_JSON]
#     --total-inferences N Max doctor-patient turns per case    [TOTAL_INFERENCES]
#     --per-call-sleep  F  Sleep between dialogue steps         [PER_CALL_SLEEP]
#     --max-parallel    N  Max concurrent case processes        [MAX_PARALLEL]
#     --in-case-concurrency N  Threads inside one case          [IN_CASE_CONCURRENCY]
#     --out-dir   PATH     Root output dir                      [OUT_DIR]
#     --grade              Pass --grade_correctness to Python    [GRADE=1]
#     --dry-run            Print commands without running        [DRY_RUN=1]
#     --                   Everything after is passed verbatim to the Python CLI
#
# EXAMPLES
#   # Two models, a few cases, 10 runs, 6 processes at a time:
#   bash generate_diagnosis_distribution/run_distribution.sh \
#     --models "gpt-5.5,gpt-5.4-mini" --cases "2,5,15,18,23-26,28" \
#     --runs 10 --max-parallel 6
#
#   # Single model, temperature sweep, semantic bucketing:
#   bash generate_diagnosis_distribution/run_distribution.sh \
#     --models deepseek-v4-pro --cases 0-9 --temps "0,0.05,0.7" \
#     --bucketing semantic --grade
#
#   # Compare prompt styles (each style runs as its own isolated process):
#   bash generate_diagnosis_distribution/run_distribution.sh \
#     --models gpt-5.5 --cases 0-9 \
#     --styles "default,conservative_safety_first,aggressive_efficiency_first"
#   # ...or every style defined in the prompt JSON:
#   bash generate_diagnosis_distribution/run_distribution.sh \
#     --models gpt-5.5 --cases 0-9 --styles all
#
#   # Pass extra raw flags through to the Python script:
#   bash generate_diagnosis_distribution/run_distribution.sh \
#     --models gpt-5.5 --cases 0-4 -- --verbose
# =============================================================================
set -uo pipefail

# ----------------------------------------------------------------------------
# Resolve paths: run everything from the repo root (parent of this script dir).
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY_SCRIPT="generate_diagnosis_distribution/diagnosis_distribution.py"
PYTHON="${PYTHON:-python}"

# ----------------------------------------------------------------------------
# Defaults (override via env var or flag).
# ----------------------------------------------------------------------------
MODELS="${MODELS:-deepseek-v4-pro}"
CASES="${CASES:-0}"
RUNS="${RUNS:-10}"
DATASET="${DATASET:-MedQA}"
TEMPS="${TEMPS:-0.05}"
PATIENT="${PATIENT:-deepseek-v4-flash}"
MEASUREMENT="${MEASUREMENT:-deepseek-v4-pro}"
MODERATOR="${MODERATOR:-deepseek-v4-pro}"
BUCKETING="${BUCKETING:-exact}"
STYLES="${STYLES:-default}"
PROMPT_JSON="${PROMPT_JSON:-}"
TOTAL_INFERENCES="${TOTAL_INFERENCES:-20}"
PER_CALL_SLEEP="${PER_CALL_SLEEP:-0.5}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
IN_CASE_CONCURRENCY="${IN_CASE_CONCURRENCY:-1}"
OUT_DIR="${OUT_DIR:-results/dist_$(date +%Y%m%d_%H%M%S)}"
GRADE="${GRADE:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()

# ----------------------------------------------------------------------------
# Parse flags.
# ----------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --models)              MODELS="$2"; shift 2 ;;
    --cases)               CASES="$2"; shift 2 ;;
    --runs)                RUNS="$2"; shift 2 ;;
    --dataset)             DATASET="$2"; shift 2 ;;
    --temps)               TEMPS="$2"; shift 2 ;;
    --patient)             PATIENT="$2"; shift 2 ;;
    --measurement)         MEASUREMENT="$2"; shift 2 ;;
    --moderator)           MODERATOR="$2"; shift 2 ;;
    --bucketing)           BUCKETING="$2"; shift 2 ;;
    --styles)              STYLES="$2"; shift 2 ;;
    --prompt-json)         PROMPT_JSON="$2"; shift 2 ;;
    --total-inferences)    TOTAL_INFERENCES="$2"; shift 2 ;;
    --per-call-sleep)      PER_CALL_SLEEP="$2"; shift 2 ;;
    --max-parallel)        MAX_PARALLEL="$2"; shift 2 ;;
    --in-case-concurrency) IN_CASE_CONCURRENCY="$2"; shift 2 ;;
    --out-dir)             OUT_DIR="$2"; shift 2 ;;
    --grade)               GRADE=1; shift ;;
    --dry-run)             DRY_RUN=1; shift ;;
    --)                    shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help)             sed -n '2,68p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------
# Expand "2,5,7-9" -> "2 5 7 8 9"; split "a,b" -> array.
# ----------------------------------------------------------------------------
expand_cases() {
  local spec="$1" part a b i
  for part in ${spec//,/ }; do
    if [[ "$part" == *-* ]]; then
      a="${part%-*}"; b="${part#*-}"
      for ((i=a; i<=b; i++)); do echo "$i"; done
    else
      echo "$part"
    fi
  done
}
slug() { echo "$1" | tr '/ ' '__'; }

# Expand styles: a comma list, or "all" (read every key from the prompt JSON).
expand_styles() {
  local spec="$1"
  if [ "$spec" = "all" ]; then
    "$PYTHON" - "$PROMPT_JSON" <<'PY'
import json, os, sys
p = sys.argv[1] or "doctor_prompts.json"
if not os.path.isabs(p):
    p = os.path.join("AgentClinic", p)
with open(p, encoding="utf-8") as f:
    text = "\n".join(l for l in f if not l.lstrip().startswith("//"))
for k in json.loads(text):
    print(k)
PY
  else
    echo "${spec//,/ }"
  fi
}

mapfile -t CASE_LIST < <(expand_cases "$CASES")
mapfile -t STYLE_LIST < <(expand_styles "$STYLES")
IFS=',' read -r -a MODEL_LIST <<< "${MODELS// /}"

if [ "${#CASE_LIST[@]}" -eq 0 ] || [ "${#MODEL_LIST[@]}" -eq 0 ] || [ "${#STYLE_LIST[@]}" -eq 0 ]; then
  echo "ERROR: empty --models, --cases, or --styles" >&2; exit 2
fi

# ----------------------------------------------------------------------------
# Banner.
# ----------------------------------------------------------------------------
echo "================================================================"
echo " run_distribution.sh  |  $(date)"
echo " repo root         : $REPO_ROOT"
echo " python            : $PYTHON"
echo " models            : ${MODEL_LIST[*]}"
echo " cases             : ${CASE_LIST[*]}"
echo " runs/case         : $RUNS     dataset: $DATASET    temps: $TEMPS"
echo " patient/meas/mod  : $PATIENT / $MEASUREMENT / $MODERATOR"
echo " bucketing         : $BUCKETING   grade: $GRADE"
echo " prompt styles     : ${STYLE_LIST[*]}${PROMPT_JSON:+   (json: $PROMPT_JSON)}"
echo " max parallel procs: $MAX_PARALLEL   in-case concurrency: $IN_CASE_CONCURRENCY"
echo " out dir           : $OUT_DIR"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo " extra args        : ${EXTRA_ARGS[*]}"
echo "================================================================"

# ----------------------------------------------------------------------------
# Throttle: block until fewer than MAX_PARALLEL background jobs are running.
# ----------------------------------------------------------------------------
wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n 2>/dev/null || true
  done
}

# ----------------------------------------------------------------------------
# Launch one process per (model, case), throttled.
# ----------------------------------------------------------------------------
run_one() {
  local model="$1" style="$2" case_id="$3"
  local mslug; mslug="$(slug "$model")"
  local case_out="$OUT_DIR/$mslug/$style/case$case_id"
  local log="$OUT_DIR/$mslug/$style/case$case_id.log"
  mkdir -p "$case_out"

  local cmd=(
    "$PYTHON" -u "$PY_SCRIPT"
    --dataset "$DATASET" --scenario_ids "$case_id" --runs "$RUNS"
    --doctor_llm "$model" --patient_llm "$PATIENT"
    --measurement_llm "$MEASUREMENT" --moderator_llm "$MODERATOR"
    --bucketing "$BUCKETING" --temperatures "$TEMPS"
    --doctor_prompt_style "$style"
    --total_inferences "$TOTAL_INFERENCES" --per_call_sleep "$PER_CALL_SLEEP"
    --concurrency "$IN_CASE_CONCURRENCY" --out_dir "$case_out"
  )
  [ -n "$PROMPT_JSON" ] && cmd+=(--doctor_prompt_json "$PROMPT_JSON")
  [ "$GRADE" = "1" ] && cmd+=(--grade_correctness)
  [ "${#EXTRA_ARGS[@]}" -gt 0 ] && cmd+=("${EXTRA_ARGS[@]}")

  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] ${cmd[*]}  > $log"
    return 0
  fi

  # Subshell records its own exit code so we can tally failures after `wait`.
  (
    "${cmd[@]}" > "$log" 2>&1
    echo "$?" > "$case_out/.status"
  ) &
  echo "  launched  model=$model style=$style case=$case_id  (pid $!)  -> $log"
}

mkdir -p "$OUT_DIR"
for model in "${MODEL_LIST[@]}"; do
  for style in "${STYLE_LIST[@]}"; do
    for case_id in "${CASE_LIST[@]}"; do
      wait_for_slot
      run_one "$model" "$style" "$case_id"
    done
  done
done
wait

[ "$DRY_RUN" = "1" ] && { echo "dry-run complete."; exit 0; }

# ----------------------------------------------------------------------------
# Tally failures from per-case .status files.
# ----------------------------------------------------------------------------
echo "================================================================"
fail=0
for model in "${MODEL_LIST[@]}"; do
  mslug="$(slug "$model")"
  for style in "${STYLE_LIST[@]}"; do
    for case_id in "${CASE_LIST[@]}"; do
      st_file="$OUT_DIR/$mslug/$style/case$case_id/.status"
      st="$(cat "$st_file" 2>/dev/null || echo "missing")"
      if [ "$st" != "0" ]; then
        echo "FAILED  model=$model style=$style case=$case_id  (status=$st, see $OUT_DIR/$mslug/$style/case$case_id.log)"
        fail=1
      fi
    done
  done
done

# ----------------------------------------------------------------------------
# Assemble a clean tree:
#   <OUT_DIR>/assembled/<model>/<style>/temp_<T>/case_<id>.json
# ----------------------------------------------------------------------------
echo "================ ASSEMBLING ================"
ASM="$OUT_DIR/assembled"
rm -rf "$ASM"
IFS=',' read -r -a TEMP_LIST <<< "${TEMPS// /}"
missing=0
for model in "${MODEL_LIST[@]}"; do
  mslug="$(slug "$model")"
  for style in "${STYLE_LIST[@]}"; do
    for t in "${TEMP_LIST[@]}"; do
      dest="$ASM/$mslug/$style/temp_$t"; mkdir -p "$dest"
      for case_id in "${CASE_LIST[@]}"; do
        src="$OUT_DIR/$mslug/$style/case$case_id/temp_$t/case_$case_id.json"
        if [ -f "$src" ]; then
          cp "$src" "$dest/case_$case_id.json"
        else
          echo "MISSING: $src"; missing=1
        fi
      done
    done
  done
done

echo "================================================================"
[ "$fail" -ne 0 ]    && echo "WARNING: some cases FAILED (see FAILED lines above)."
[ "$missing" -ne 0 ] && echo "WARNING: some result files were MISSING (see MISSING lines above)."
[ "$fail" -eq 0 ] && [ "$missing" -eq 0 ] && echo "ALL OK."
echo "assembled tree -> $ASM"
echo "DONE: $(date)"
exit "$fail"
