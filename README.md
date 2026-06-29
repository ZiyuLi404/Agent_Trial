# Version-Aware Clinical Trial

A prototype that wraps [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic) with sequential case streaming, version epoch tracking, 1:1 concurrent-control randomization, and JSONL trial logging.

AgentClinic is the simulation engine. This repo adds a thin trial layer on top — the doctor agent is treated as a versioned, time-varying intervention. Each substantial update opens a new epoch. Epochs are evaluated separately using concurrent control.

---

## 📁 Module Layout (refactor in progress)

The code is split into modules, **each with its own `README.md`**. The full blueprint is in [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md).

| Module | Role |
|--------|------|
| [`AgentClinic/`](AgentClinic/README.md) | Simulation engine (the foundation) |
| [`trial/`](trial/README.md) | Online trial framework |
| [`anchor_compare/`](anchor_compare/README.md) | Anchored regression (behavioral drift) |
| [`deployment_replay/`](deployment_replay/README.md) | Deployment replay / hybrid estimation |
| [`generate_diagnosis_distribution/`](generate_diagnosis_distribution/README.md) | Diagnosis-distribution generator (upstream producer) |
| [`embedding_similarity/`](embedding_similarity/README.md) | Drift · Method B (vectors + similarity) |
| [`kl_js_divergence/`](kl_js_divergence/README.md) | Drift · Method A (distributions + divergence) |
| [`history_borrowing/`](history_borrowing/README.md) | History borrowing / performance estimation |
| [`lora_fingerprint/`](lora_fingerprint/README.md) | Model/version fingerprint detection (DistilBERT, optional LoRA) |

> ⚠️ **Commands moved**: scripts now live inside module dirs — run them from the **repo root**, e.g. `python trial/run_trial.py ...` (or `python -m trial.run_trial ...`).

---

## Control Strategy

A single frozen control model (`deepseek-v4-flash` by default, set via `CONTROL_MODEL` in `trial/run_trial.py`) is maintained across all epochs. Treatment versions change sequentially. Each treatment version is evaluated against concurrent control cases from the same time window — not against a version-specific control pool.

| Role | Configured by | Changes over time? |
|---|---|---|
| Control doctor | `--control_llm` (default: `deepseek-v4-flash`) | No — frozen for all epochs |
| Treatment doctor | `--doctor_llm` | Yes — updates with each `--new_version` call |
| Patient / Measurement / Moderator | `--patient_llm` / `--measurement_llm` / `--moderator_llm` | Shared across both arms |

---

## Phase 1 Trial Flow

```
case arrives → version_manager marks active epoch
             → randomizer assigns control / treatment
             → control arm → frozen CONTROL_MODEL
               treatment arm → active --doctor_llm
             → AgentClinic runs the dialogue
             → logger appends result to trial_log.jsonl
             → repeat
```

Per-epoch treatment effects are computed offline from `trial_log.jsonl` after the trial.

---

## Environment Setup

```bash
# Create environment
conda create -n agenttrial python=3.10
conda activate agenttrial

# Install dependencies
pip install -r requirements.txt
```

### API keys (`.env` file — recommended for team collaboration)

Copy the template and fill in your own keys:

```bash
cp .env.example .env
# edit .env with your real keys
```

`.env` is gitignored, so each collaborator keeps their own keys locally. Both `trial/run_trial.py` and `anchor_compare/anchor_compare.py` call `python-dotenv` at startup, so anything in `.env` is automatically loaded into the process environment — no need to `source` or `export` manually.

To add a new provider, append a line like `NEW_PROVIDER_API_KEY=...` to both `.env` (your real key) and `.env.example` (empty placeholder, committed for the team).

Different models require different keys: DeepSeek models need `DEEPSEEK_API_KEY`; GPT models need `OPENAI_API_KEY`; Claude models need `ANTHROPIC_API_KEY`; Llama / Mixtral on Replicate need `REPLICATE_API_TOKEN`.

If you prefer not to use `.env`, you can still `export` the variables in your shell or pass them as CLI flags (`--deepseek_api_key`, `--openai_api_key`, etc.) — but CLI flags get recorded in shell history and are visible in `ps`, so avoid them for real keys.

---

## Running Trials

### Start version v1 — treatment: deepseek-v4-flash, control: deepseek-v4-flash (frozen)

> v1 intentionally uses the same model for treatment and control. This is a **calibration epoch** that verifies the trial pipeline produces near-identical results when no real difference exists, before introducing a real treatment change in v2.

```bash
python trial/run_trial.py \
  --new_version --version_id v1 --model_name deepseek-v4-flash \
  --prompt_version p1 --tool_version t1 \
  --deepseek_api_key $DEEPSEEK_API_KEY \
  --control_llm deepseek-v4-flash \
  --doctor_llm deepseek-v4-flash \
  --patient_llm deepseek-v4-flash \
  --measurement_llm deepseek-v4-flash \
  --moderator_llm deepseek-v4-flash \
  --dataset MedQA --num_cases 20 --total_inferences 20
```

### Continue under the same version

```bash
python trial/run_trial.py \
  --deepseek_api_key $DEEPSEEK_API_KEY \
  --control_llm deepseek-v4-flash \
  --doctor_llm deepseek-v4-flash \
  --dataset MedQA --num_cases 20
```

### Upgrade to v2 — treatment: deepseek-v4-pro, control unchanged

```bash
python trial/run_trial.py \
  --new_version --version_id v2 --model_name deepseek-v4-pro \
  --prompt_version p1 --tool_version t1 \
  --deepseek_api_key $DEEPSEEK_API_KEY \
  --control_llm deepseek-v4-flash \
  --doctor_llm deepseek-v4-pro \
  --patient_llm deepseek-v4-flash \
  --measurement_llm deepseek-v4-flash \
  --moderator_llm deepseek-v4-flash \
  --dataset MedQA --num_cases 20 --total_inferences 20
```

Each call appends to `trial_log.jsonl`. Every record is tagged with `version_id`, `arm`, `control_model`, and `treatment_model`.

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--control_llm` | `deepseek-v4-flash` | Frozen control doctor model (fixed across all epochs) |
| `--doctor_llm` | `deepseek-v4-pro` | Treatment doctor model (changes with each version) |
| `--patient_llm` | `deepseek-v4-flash` | Patient agent model |
| `--measurement_llm` | `deepseek-v4-flash` | Measurement agent model |
| `--moderator_llm` | `deepseek-v4-flash` | Moderator / judge model |
| `--dataset` | `MedQA` | `MedQA`, `MedQA_Ext`, `NEJM`, `NEJM_Ext` |
| `--num_cases` | all | Cases to run |
| `--total_inferences` | `20` | Max doctor–patient turns per case |
| `--new_version` | off | Open a new version epoch |
| `--version_id` | `v1` | Version label |
| `--model_name` | `""` | Human-readable model label |
| `--prompt_version` | `p1` | Prompt version label |
| `--tool_version` | `t1` | Tool version label |
| `--openai_api_key` | env | OpenAI API key |
| `--deepseek_api_key` | env | DeepSeek API key |

---

## Output and Logs

**`trial_log.jsonl`** — append-only, one JSON object per line:

```json
{
  "case_id": 0,
  "timestamp": "2026-05-18T14:03:22.418Z",
  "version_id": "v1",
  "arm": "control",
  "control_model": "deepseek-v4-flash",
  "treatment_version": "v1",
  "treatment_model": "deepseek-v4-flash",
  "diagnosis": "DIAGNOSIS READY: Pneumonia",
  "correct_diagnosis": "Pneumonia",
  "correctness": true,
  "confidence": null,
  "compliance": null,
  "consultation": "Doctor: ...\nPatient: ...\n"
}
```

`confidence` and `compliance` are reserved for future phases and logged as `null` in Phase 1.

**`current_version.json`** — persists the active version epoch:

```json
{
  "version_id": "v2",
  "model_name": "deepseek-v4-flash",
  "prompt_version": "p1",
  "tool_version": "t1"
}
```

---

## Supported Models

| Model | Provider |
|---|---|
| `deepseek-v4-flash`, `deepseek-v4-pro` (and legacy aliases `deepseek-chat`, `deepseek-reasoner`) | DeepSeek |
| `gpt4`, `gpt4o`, `gpt-4o-mini`, `gpt3.5`, `o1-preview` | OpenAI |
| `claude3.5sonnet` | Anthropic |
| `llama-2-70b-chat`, `llama-3-70b-instruct`, `mixtral-8x7b` | Replicate |

---

## Code Structure

See the [Module Layout](#-module-layout-refactor-in-progress) table above; each module
documents its own files in its `README.md`. The online-trial code (this section's
original focus) now lives in [`trial/`](trial/README.md):

| File | Role |
|---|---|
| `trial/run_trial.py` | CLI entry point — orchestrates version management, streaming, logging, and accuracy display |
| `trial/trial_manager.py` | Sequential case streaming (`stream_cases`) and case execution wrapper (`run_case`) |
| `trial/version_manager.py` | Creates and persists version epoch state |
| `trial/randomization.py` | 1:1 block randomization — assigns `control` or `treatment` per case |
| `trial/logger.py` | Appends per-case records to `trial_log.jsonl` |
| `AgentClinic/agentclinic.py` | Core simulation engine: DoctorAgent, PatientAgent, MeasurementAgent, ScenarioLoaders |
| `AgentClinic/agentclinic_*.jsonl` | MedQA / NEJM scenario datasets (+ extended) |

---

## Roadmap

The original roadmap, mapped to what now exists. Done items have become their own modules.

### ✅ Done
- **Historical borrowing** — [`history_borrowing/`](history_borrowing/README.md) borrows accuracy across *similar models*; [`deployment_replay/`](deployment_replay/README.md) borrows a version's *own past cases* via transcript replay.
- **Drift detection (diagnostic distribution / model behavior)** — [`embedding_similarity/`](embedding_similarity/README.md) (Method B, vectors), [`kl_js_divergence/`](kl_js_divergence/README.md) (Method A, ICD distributions), and [`anchor_compare/`](anchor_compare/README.md) (behavioral drift on fixed anchors).

### 🟡 In progress / partial
- **Parallel / concurrent execution** — `generate_diagnosis_distribution` has `--concurrency`; the online trial loop itself is still sequential (no thread-safe logging / version management yet).
- **Improved analysis layer** — similarity / divergence / borrowing are covered by G·H·I; per-version treatment-effect estimation, sensitivity analysis, and epoch-comparison reports are not built. Plotting utilities live in `figures_and_reports/` (archived).

### ❌ To do
- **Auto-trigger epochs on drift** — detect case-mix / tool-usage shifts and open a new version automatically (the removed `version_detect.py` was an early attempt).
- **Expanded outcomes** — patient confidence, compliance, consultation willingness, safety / bias metrics, tool-use statistics (today `confidence` / `compliance` are logged as `null`).