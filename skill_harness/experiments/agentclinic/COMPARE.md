# AgentClinic variant experiment

This experiment is separate from `AgentClinic/` and the stable
`trial/run_trial.py` CLI. The external adapter applies a versioned skill, a
versioned harness, or both, and compares that variant against the unchanged
AgentClinic baseline on the same cases.

Skill-only dry run:

```bash
python -m skill_harness.experiments.agentclinic.compare \
  --cases 0 \
  --skill_path skill_harness/artifacts/seeds/diagnostic_reasoning/v000.md \
  --dry_run
```

Harness-only dry run:

```bash
python -m skill_harness.experiments.agentclinic.compare \
  --cases 0 \
  --harness_path skill_harness/artifacts/harnesses/baseline/diagnostic_efficiency_v000.toml \
  --dry_run
```

Combined skill+harness live comparison:

```bash
python -m skill_harness.experiments.agentclinic.compare \
  --doctor_llm deepseek-v4-pro \
  --cases 0-19 \
  --skill_path skill_harness/artifacts/seeds/diagnostic_reasoning/v000.md \
  --harness_path skill_harness/artifacts/harnesses/baseline/diagnostic_efficiency_v000.toml \
  --output_dir skill_harness/results/variant_experiments/agentclinic/combined_v000
```
