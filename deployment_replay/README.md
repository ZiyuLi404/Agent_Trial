# deployment_replay — Deployment Replay / Hybrid Estimation (E)

**The problem: after upgrading to a new doctor V2, V2 only sees new patients going forward — but you have a backlog of old cases, and the old patients are gone. How do you estimate V2's performance on them?**

## Three ways to "make up" the old cases
| Way | How | Possible in reality? |
|-----|-----|----------------------|
| **Oracle** | V2 re-interviews each old patient from scratch | ❌ Only in simulation (counterfactual truth) |
| **Replay** | V2 reads the old transcript and diagnoses from it | ✅ Yes (records are on file) |
| **Hybrid** | Replay for old cases + live for new cases | ✅ Yes |

**Core question:** how much accuracy do you lose by using replay (the only thing possible in real life) instead of oracle (ideal but impossible)?
- replay ≈ oracle → you can backfill a new version from transcripts quickly, no waiting for new patients.
- replay ≪ oracle → reading transcripts loses too much (V2 would have asked different questions / ordered different tests).

## Files
| File | Purpose |
|------|---------|
| `deployment_timeline.py` | Entry point — runs the day-by-day timeline and ties the three together |
| `replay_evaluator.py` | Replay: V2 diagnoses from the saved transcript only |
| `oracle_evaluator.py` | Oracle: V2 re-interviews from scratch (reuses `trial.trial_manager.run_case`) |
| `hybrid_estimator.py` | Combines replay (old) + live (new), weighted by case counts |

## Run
```bash
python -m deployment_replay.deployment_timeline ...
python deployment_replay/deployment_timeline.py ...
```

## Relationship
- **Depends on** `trial` (`from trial.trial_manager import ...`) and `AgentClinic`.
- Sibling of `history_borrowing`: both answer "a new version has too little data — how do we borrow from the past?" E borrows *vertically* (its own old cases, via replay), H borrows *horizontally* (similar models). Part of the "borrow" line in `notes/design.md`.

## Status
- ⚠️ **Early prototype, currently idle.** Added 2026-05-28, refreshed 2026-06-08, not advanced since; **never produced output in this tree** (no `deployment_log.jsonl` / `transcripts/`). Kept, not dead.
- ✅ Moved into the `deployment_replay/` package; imports switched to absolute `trial.*` / `deployment_replay.*`.
- Outputs will land in `results/deployment_replay/` (see `REFACTOR_PLAN.md`).
