# anchor_compare — Anchored Regression (C)

**Tells whether a version upgrade changed the *model's behavior* or just got *harder cases*. It does NOT use the gold answer.**

## The idea
Pick a fixed set of "anchor" cases. Run both the baseline model and the candidate model on the **same** cases, several times each. Since the cases are identical, any difference in output must come from the model itself — not from case difficulty.

## The 4 drift metrics (all gold-free)
| Metric | What it checks |
|--------|----------------|
| Top-1 agreement | Did the final diagnosis change (optionally judged "clinically equal" by a moderator) |
| Candidate Jaccard | Did the differential-diagnosis shortlist change |
| Evidence Jaccard | Did the key evidence the model focuses on change |
| JS divergence | Drift in the distribution of diagnoses across repeated runs |

To enable comparison, the DoctorAgent emits a structured output under `output_format="anchor_compare"`: `DIAGNOSIS READY` + `CANDIDATES:` + `KEY EVIDENCE:`.

## Run
```bash
python anchor_compare/anchor_compare.py --eval_mode anchor_compare \
    --baseline_doctor_llm deepseek-v4-flash \
    --candidate_doctor_llm deepseek-v4-pro \
    --num_scenarios 10 --runs_per_case 3 ...
```

## Relationship
- A second, orthogonal evaluation channel next to `trial`'s online accuracy (see `notes/design.md`): read them together to separate "cases got harder" from "model regressed."
- Same goal as `embedding_similarity` / `kl_js_divergence` (measure drift without gold) — those are the more developed offline versions.

## Status & known debt
- ⚠️ **Early prototype, currently idle.** Created 2026-05-27 and barely touched since; focus shifted to G/I/H. Kept, not dead.
- ⚠️ **It carries its own copy of the engine** (~1338 lines: its own `query_model`, agents, loaders — it does **not** `import AgentClinic`). Its `query_model` is also an old version (`temperature=0`, `max_tokens=200`, which can truncate the structured output).
  - Intended fix: import from `AgentClinic.agentclinic` and keep only a thin subclass for the `anchor_compare` structured output.
  - **Deferred by the owner for now.** See `REFACTOR_PLAN.md` §6.
- ✅ Moved into this module; `DATA_DIR` fixed to point at the repo-root `AgentClinic/` (datasets still found).
