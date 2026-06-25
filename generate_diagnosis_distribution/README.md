# generate_diagnosis_distribution — Diagnosis Distribution Generator (F · upstream producer)

**Fix one case, run the same consultation many times, and collect every final diagnosis to approximate the "diagnosis probability distribution" for that case — i.e. how stable/spread the model's output is.**

This is the **upstream producer**: the `case_X.json` files it writes are the input for **`embedding_similarity` (G)** and **`kl_js_divergence` (I)**.

## It's really a general runner (two axes)
`diagnosis_distribution.py` can sweep over `model × prompt-style`:
- Fix the prompt, look at output stability → the "diagnosis distribution" experiment.
- Vary `--doctor_prompt_style`, look at how diagnoses shift → the **prompt experiment** (prompt bank lives in `AgentClinic/doctor_prompts.json`, 5 styles, all free-text output: `default` + 4 diagnostic-style ablations that differ from `default` by a single inserted sentence).

## Files
| File | Purpose |
|------|---------|
| `diagnosis_distribution.py` | Main runner. Reuses the engine (`import AgentClinic.agentclinic`), no copy |
| `make_dist_report.py` | Renders results into a plain-text report |
| `run_distribution.sh` | General batch runner: one process per (model × style × case), throttled concurrency, isolated out dirs, assembles a clean `assembled/<model>/<style>/temp_<T>/case_<id>.json` tree |
| `run_gpt_data.sh` | Legacy GPT-only batch helper (superseded by `run_distribution.sh`) |

## Output
Each `case_X.json` has `scenario_id`, `runs`, `distribution` (per-bucket counts), `samples` (each run's `diagnosis_text` + `full_dialogue`), `entropy_bits`, etc.

## Run
```bash
# from the repo root
python generate_diagnosis_distribution/diagnosis_distribution.py \
    --doctor_llm deepseek-v4-pro --scenario_ids 0 --runs 10 ...
# multiple models in one sweep (comma-separated; GPT is just another value):
python generate_diagnosis_distribution/diagnosis_distribution.py --doctor_llm deepseek-v4-flash,deepseek-v4-pro ...
```

## Notes
- ✅ Moved into this module; the bootstrap now adds the **repo root** to `sys.path` (so `import AgentClinic` works from a subdir).
- ✅ `make_dist_report.py` reclaimed from the archive; its `RESULTS_DIR` points at the repo-root `results/`.
- Target data location: `results/generate_diagnosis_distribution/<model>/case_X.json` (this replaces the hand-zipped `Data*.zip`).
- ✅ `run_distribution.sh` is the current general runner (configurable models/cases/styles, capped concurrency, no zip step). `run_gpt_data.sh` is the legacy GPT-only script kept for reference.
