# Skill and harness experiments

This folder is a sidecar research area. Nothing under `AgentClinic/`, `trial/`,
or the other original project packages is modified by these experiments.

```text
skill_harness/
├── artifacts/                  versioned skills and harness configurations
├── common/                     shared AgentClinic adapter and result contract
├── methods/
│   ├── skillopt_original/      Microsoft SkillOpt/ReflACT; skill-only
│   ├── skillopt_lite/          coding-agent loop; skill.md-only
│   └── harnessopt/             coding-agent loop; isolated Python allowlist
├── experiments/agentclinic/    fixed manifests and comparison entry points
├── upstream/                   pinned versions; local checkouts are ignored
├── cluster/carc/               Slurm launch files (no credentials stored)
├── tests/                      sidecar-only tests
└── results/                    ignored outputs for new runs
```

## Method boundaries

- **Original SkillOpt** calls the pinned, unmodified upstream trainer and
  registers the AgentClinic environment at runtime. It optimizes the skill only;
  no Lite or HarnessOpt code is imported.
- **SkillOpt-Lite** is the paper's coding-agent loop. Its editable object is one
  workspace `skill.md`; its evaluator and paired validation gate live in its own
  module.
- **HarnessOpt** is also coding-agent-driven, but its editable surface is only
  the four Python files recorded in `methods/harnessopt/allowlist.json`. The
  skill and original AgentClinic source are read-only.

The baseline uses AgentClinic's original generative measurement agent. No
measurement or harness ablation from the discarded experiments is retained.

## Paper code

`upstream/versions.json` pins both repositories. Existing sibling checkouts are
used when available; otherwise clone local read-only checkouts with:

```bash
python -m skill_harness.upstream.sync
```

The runner rejects the wrong commit or a dirty upstream tree, preventing local
adapter changes from silently becoming part of a paper reproduction.

## Entry points

Original SkillOpt one-epoch run:

```bash
python -m skill_harness.methods.skillopt_original.run
```

SkillOpt-Lite contract test:

```bash
python -m skill_harness.methods.skillopt_lite.evaluator \
  --skill skill_harness/artifacts/seeds/diagnostic_reasoning/initial_blank.md \
  --split train --eval_limit 2 --limit 2 --contract_dry_run
```

HarnessOpt contract test:

```bash
python -m skill_harness.methods.harnessopt.evaluator \
  --split train --eval_limit 2 --limit 2 --contract_dry_run
```

New outputs go under `skill_harness/results/`. The previous skill experiments,
manual seed, split files, and run outputs were deleted before `pure_v1` was
created.
