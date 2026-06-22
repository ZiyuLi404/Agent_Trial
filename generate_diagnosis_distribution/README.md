# generate_diagnosis_distribution — Diagnosis Distribution Generator (F · upstream producer)

**Fix one case, run the same consultation many times, and collect every final diagnosis to approximate the "diagnosis probability distribution" for that case — i.e. how stable/spread the model's output is.**

This is the **upstream producer**: the `case_X.json` files it writes are the input for **`embedding_similarity` (G)** and **`kl_js_divergence` (I)**.

## It's really a general runner (two axes)
`diagnosis_distribution.py` can sweep over `model × prompt-style`:
- Fix the prompt, look at output stability → the "diagnosis distribution" experiment.
- Vary `--doctor_prompt_style`, look at how diagnoses shift → the **prompt experiment** (prompt bank lives in `AgentClinic/doctor_prompts.json`, 7 styles).

## Files
| File | Purpose |
|------|---------|
| `diagnosis_distribution.py` | Main runner. Reuses the engine (`import AgentClinic.agentclinic`), no copy |
| `make_dist_report.py` | Renders results into a plain-text report |
| `run_gpt_data.sh` | Batch helper: runs per case, assembles `<group>/case_X.json` folders |

## Output
Each `case_X.json` has `scenario_id`, `runs`, `distribution` (per-bucket counts), `samples` (each run's `diagnosis_text` + `full_dialogue`), `entropy_bits`, etc.

## Run
```bash
# from the repo root
python generate_diagnosis_distribution/diagnosis_distribution.py \
    --doctor_llm deepseek-v4-flash --scenario_ids 0,1,2 --runs 30 ...
# multiple models in one sweep (GPT is just another model value):
python generate_diagnosis_distribution/diagnosis_distribution.py --doctor_llm gpt-5.5,gpt-5-mini ...
```

## Notes
- ✅ Moved into this module; the bootstrap now adds the **repo root** to `sys.path` (so `import AgentClinic` works from a subdir).
- ✅ `make_dist_report.py` reclaimed from the archive; its `RESULTS_DIR` points at the repo-root `results/`.
- Target data location: `results/generate_diagnosis_distribution/<model>/case_X.json` (this replaces the hand-zipped `Data*.zip`).
- ⚠️ The trailing `zip -r Data_gpt.zip` step in `run_gpt_data.sh` is **obsolete** now that zips are retired — drop that step; keep the orchestration.
