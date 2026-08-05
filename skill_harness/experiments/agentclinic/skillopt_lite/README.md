# AgentClinic × SkillOpt-Lite

SkillOpt-Lite is a coding-agent-driven, skill-only optimizer. It may edit only
the workspace `skill.md`; AgentClinic code, harness code, evaluator, and split
manifest are fixed.

The new `pure_v1` protocol uses a fresh seed (`20260804`) to split all 107
MedQA cases into 21 train, 21 validation, and 65 test cases. It replaces every
previous split. Validation trajectories must not be inspected or used to write
the next skill; test runs once after the best validation version is selected.

Safe contract run from the repository root:

```bash
python -m skill_harness.methods.skillopt_lite.evaluator \
  --skill skill_harness/artifacts/seeds/diagnostic_reasoning/initial_blank.md \
  --manifest skill_harness/experiments/agentclinic/manifests/medqa_pure_v1.json \
  --split train --eval_limit 2 --limit 2 --contract_dry_run
```

Live outputs are written beneath the ignored
`skill_harness/results/skillopt_workspaces/` directory. Each run contains the
frozen manifest, result rows, metrics, trajectories, backend health, and sample
Markdown files. Use `skill_harness.methods.skillopt_lite.gate` to compare
baseline and candidate results on identical validation IDs.
