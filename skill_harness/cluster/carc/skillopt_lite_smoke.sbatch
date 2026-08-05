#!/bin/bash
#SBATCH --job-name=ac-lite-smoke
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=skill_harness/results/carc/%x-%j.out

set -euo pipefail
: "${AGENT_TRIAL_REPO:?Set AGENT_TRIAL_REPO to the Agent_Trial_refactor checkout}"
cd "$AGENT_TRIAL_REPO"
python -m skill_harness.methods.skillopt_lite.evaluator \
  --skill skill_harness/artifacts/seeds/diagnostic_reasoning/v000.md \
  --split train --eval_limit 5 --limit 5
