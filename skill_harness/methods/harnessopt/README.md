# HarnessOpt module

This is the coding-agent-driven HarnessOpt path from SkillOpt-Lite. It is not
part of original SkillOpt/ReflACT. Its editable surface is physically confined
to `workspace/agentclinic/`; `allowlist.json` records the exact four files.

Dry-run the evaluator from the project root:

```bash
python -m skill_harness.methods.harnessopt.evaluator \
  --split train --eval_limit 2 --contract_dry_run
```

The optimizer follows `prompts/harnessopt_loop.md`, examines generated samples,
edits only the allowlist, and uses validation-gated archive/restore. The
baseline workspace delegates to the unchanged AgentClinic behavior, so round 0
is a genuine no-harness-change control.
