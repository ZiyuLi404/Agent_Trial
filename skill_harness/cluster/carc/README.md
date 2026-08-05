# CARC launch notes

These files contain no API key and do not submit anything automatically.

On a CARC login node, place or clone the project in scratch, create the output
directory, activate the Python environment, and export the repository path:

```bash
export AGENT_TRIAL_REPO=/scratch1/junhanwu/Adaptive_Agent_trail/Agent_Trial_refactor
mkdir -p "$AGENT_TRIAL_REPO/skill_harness/results/carc"
cd "$AGENT_TRIAL_REPO"
```

Provide credentials through the job environment or a permission-restricted
environment file outside the repository. Never put keys in an sbatch file.

The original SkillOpt launcher defaults to 12 isolated case processes. Change
`SKILLOPT_WORKERS` only after checking provider concurrency and rate limits.
The Lite and HarnessOpt launchers currently remain sequential because their
coding-agent loops perform validation-gated file edits between evaluations;
parallelism should be applied within an evaluation, not across competing edits.
