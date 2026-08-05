# AgentClinic HarnessOpt loop

Optimize only the Python files listed in `../allowlist.json`. The fixed skill,
AgentClinic core, evaluator, split manifest, and both SkillOpt modules are
read-only.

For each round:

1. Archive all allow-listed files under
   `skill_harness/results/harnessopt/workspace/.skillopt/history/<round>/before/`.
2. Read the latest train samples and trajectories. Classify failures as model,
   patient/measurement backend, doctor construction, or interaction-loop errors.
3. Make the smallest auditable patch to one or more allow-listed Python files.
4. Confirm no path outside the allowlist changed.
5. Run a two-case contract dry-run, then a small live smoke run.
6. Evaluate the same clean validation IDs for baseline and candidate. Promote
   only a strictly better candidate; otherwise restore the archived files.
7. Save the accepted files and score under `history/<round>/after/`.

Never edit the skill in this loop. Never interpret empty patient, measurement,
or doctor responses as clinical failures; mark them as infrastructure failures.
Run the held-out test split once, after the final validation-selected version.
