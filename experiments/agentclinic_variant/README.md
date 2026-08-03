# AgentClinic variant experiment

This experiment is separate from `AgentClinic/` and the stable
`trial/run_trial.py` CLI. The external adapter applies a versioned skill, a
versioned harness, or both, and compares that variant against the unchanged
AgentClinic baseline on the same cases.

Skill-only dry run:

```bash
python -m experiments.agentclinic_variant.compare \
  --cases 0 \
  --skill_path change_generators/skills/artifacts/diagnostic_reasoning/v000.md \
  --dry_run
```

Harness-only dry run:

```bash
python -m experiments.agentclinic_variant.compare \
  --cases 0 \
  --harness_path change_generators/harnesses/artifacts/agentclinic/diagnostic_efficiency/v000.toml \
  --dry_run
```

Combined skill+harness live comparison:

```bash
python -m experiments.agentclinic_variant.compare \
  --doctor_llm deepseek-v4-pro \
  --cases 0-19 \
  --skill_path change_generators/skills/artifacts/diagnostic_reasoning/v000.md \
  --harness_path change_generators/harnesses/artifacts/agentclinic/diagnostic_efficiency/v000.toml \
  --output_dir results/variant_experiments/agentclinic/combined_v000
```
