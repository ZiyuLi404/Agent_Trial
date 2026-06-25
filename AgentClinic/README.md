# AgentClinic — Simulation Engine (the foundation)

**A simulated clinic where an AI doctor, AI patient, and AI measurement reader talk to each other to reach a diagnosis. Every module that actually "runs a consultation" is built on this.**

Derived from upstream [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic), with a few additions of our own (see below).

## Files
| File | Purpose |
|------|---------|
| `agentclinic.py` | Core engine: the agents, `ScenarioLoader*`, `query_model`, `compare_results`, etc. |
| `doctor_prompts.json` | Doctor prompt bank (5 styles, all free-text output) — the material for the prompt experiment |
| `agentclinic_*.jsonl` (×4) | MedQA / NEJM case datasets (+ extended) |
| `icd10cm_2026.jsonl`, `icd10cm_codes/` | **The single source** of the ICD-10-CM dictionary (shared by the engine and by `kl_js_divergence`) |

## How others use it
Everything imports it as a package:
```python
from AgentClinic.agentclinic import DoctorAgent, query_model, ScenarioLoaderMedQA
```
Used by `trial`, `deployment_replay`, and `generate_diagnosis_distribution`.
(`anchor_compare` currently keeps its **own copy** of the engine instead of importing — see its README.)

## What we added on top of upstream
- `doctor_prompt_template` / `load_doctor_prompt_template` — switch doctor "personas" from `doctor_prompts.json` (backward compatible; default behaves like upstream).
- ICD-10-CM validation helpers.
- Support for GPT-5.x reasoning models and DeepSeek `reasoning_content`.

## Maintenance rule
This is the **foundation** — keep it stable and change it rarely. Prefer **additive, backward-compatible** changes so the layers above don't break. Keep only one copy of the ICD dictionary here.
