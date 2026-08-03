# AgentClinic skill experiment

This experiment is intentionally separate from `AgentClinic/` and the stable
`trial/run_trial.py` CLI. It applies a versioned skill through an external
`AgentClinicAdapter` and performs a paired no-skill/with-skill comparison.

Preview the exact prompts without API calls:

```bash
python -m experiments.agentclinic_skill.compare \
  --cases 0 \
  --skill_path change_generators/skills/artifacts/diagnostic_reasoning/v000.md \
  --output_dir results/skill_experiments/agentclinic/dry_run \
  --dry_run
```

Run a live comparison:

```bash
python -m experiments.agentclinic_skill.compare \
  --doctor_llm deepseek-v4-pro \
  --cases 0-19 \
  --skill_path change_generators/skills/artifacts/diagnostic_reasoning/v000.md \
  --output_dir results/skill_experiments/agentclinic/diagnostic_reasoning_v000
```
