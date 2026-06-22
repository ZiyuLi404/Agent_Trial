# trial — Online Trial Framework

**Treats the doctor agent as a versioned intervention (V1 → V2 → V3 …): streams cases through the engine one by one, randomly assigns each to control/treatment, and logs every result — to measure how accuracy changes when the doctor is upgraded.**

Built on top of the **`AgentClinic`** engine: the engine runs one consultation, this module organizes thousands of them into a trial.

## Files
| File | Purpose |
|------|---------|
| `run_trial.py` | CLI entry point — orchestrates versions, case streaming, logging, accuracy |
| `trial_manager.py` | `stream_cases` (sequential case feed) and `run_case` (one consultation) |
| `version_manager.py` | Read/write the version-epoch state (`current_version.json`) |
| `randomization.py` | 1:1 block randomization → `control` / `treatment` per case |
| `logger.py` | Append each result to `trial_log.jsonl`; `log_deployment_case` for `deployment_replay` |
| `current_version.json` | Current version pointer (module state; path pinned to this dir) |

## Run
```bash
# from the repo root — either form works
python -m trial.run_trial --new_version --version_id v1 --model_name deepseek-v4-flash ...
python trial/run_trial.py    --new_version --version_id v1 ...
```

## Relationship
- **Depends on** `AgentClinic` (`from AgentClinic.agentclinic import ...`).
- **Reused by** `deployment_replay` (`from trial.trial_manager import ...`).

## Notes
- Refactor: the five files moved into the `trial/` package; imports switched to absolute (`from trial.X import ...`); `current_version.json` path pinned to this directory.
- Outputs (`trial_log.jsonl`) will move to `results/trial/` later (see `REFACTOR_PLAN.md`).
- Known debt: `run_case` does not yet wire in the ReviewerAgent (the "two-doctor workflow" in the top-level README); `randomization._block` is not persisted across processes.
