# AgentClinic SkillOpt-Lite loop

Optimize only the selected workspace's `skill.md`. AgentClinic source, harness
code, split manifests, evaluators, and the original SkillOpt module are
read-only.

Start with a full clean-validation baseline and one train rollout. For each
round, archive `skill.md`, inspect the newest train failures and successes,
apply the smallest evidence-backed skill edit, and evaluate the full validation
split on identical IDs. Promote only a strictly better validation score; on a
tie or regression restore the archived skill. Generate a fresh train batch for
the next round. Run the held-out test split exactly once after selecting the
best validation version.

Backend-empty rows (`agent_ok=false`) are excluded from the clinical gate and
must be investigated as infrastructure failures, not used as optimization
signals.
