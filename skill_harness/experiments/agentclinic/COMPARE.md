# AgentClinic paired comparison

This utility compares the unchanged AgentClinic baseline against an explicitly
provided skill or harness artifact on identical case IDs. New experiments begin
with the neutral blank seed; no artifact from the discarded runs remains.

Skill-only contract check:

```bash
python -m skill_harness.experiments.agentclinic.compare \
  --cases 0 \
  --skill_path skill_harness/artifacts/seeds/diagnostic_reasoning/initial_blank.md \
  --dry_run
```

Formal SkillOpt runs should use the frozen IDs in
`manifests/medqa_pure_v1.json`; this free-form comparison utility is for
diagnostics, not validation or test reporting.
