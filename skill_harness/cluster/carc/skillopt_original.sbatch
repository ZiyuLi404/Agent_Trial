#!/bin/bash
#SBATCH --job-name=ac-skillopt
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=skill_harness/results/carc/%x-%j.out

set -euo pipefail
: "${AGENT_TRIAL_REPO:?Set AGENT_TRIAL_REPO to the Agent_Trial_refactor checkout}"
cd "$AGENT_TRIAL_REPO"
python -m skill_harness.methods.skillopt_original.run \
  --cfg-options "env.workers=${SKILLOPT_WORKERS:-12}"
