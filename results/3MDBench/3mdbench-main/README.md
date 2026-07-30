# 3MDBench DeepSeek Text-Only Run

This is a trimmed copy of the local 3MDBench run used for the
`deepseek_first100_pro_flash` experiment.

The original multimodal path was not used for the final run because the
DeepSeek chat API rejected `image_url` message parts. The saved experiment was
run with `--no_image`.

## Files Used

- `benchmarking/run_deepseek_pipeline.py`
- `utils/dialogue_utils.py`
- `requirements.txt`
- Hugging Face dataset: `univanxx/3mdbench`, split `test`

## Re-run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the first 100 cases with four shards:

```bash
export DEEPSEEK_API_KEY="your_key_here"

python3 benchmarking/run_deepseek_pipeline.py --experiment_name deepseek_first100_pro_flash --case_ids 0-24 --doctor_model deepseek-v4-pro --patient_model deepseek-v4-flash --assessment_model deepseek-v4-pro --diagnosis_model deepseek-v4-pro --max_tokens 2000 --max_turns 20 --no_image
python3 benchmarking/run_deepseek_pipeline.py --experiment_name deepseek_first100_pro_flash --case_ids 25-49 --doctor_model deepseek-v4-pro --patient_model deepseek-v4-flash --assessment_model deepseek-v4-pro --diagnosis_model deepseek-v4-pro --max_tokens 2000 --max_turns 20 --no_image
python3 benchmarking/run_deepseek_pipeline.py --experiment_name deepseek_first100_pro_flash --case_ids 50-74 --doctor_model deepseek-v4-pro --patient_model deepseek-v4-flash --assessment_model deepseek-v4-pro --diagnosis_model deepseek-v4-pro --max_tokens 2000 --max_turns 20 --no_image
python3 benchmarking/run_deepseek_pipeline.py --experiment_name deepseek_first100_pro_flash --case_ids 75-99 --doctor_model deepseek-v4-pro --patient_model deepseek-v4-flash --assessment_model deepseek-v4-pro --diagnosis_model deepseek-v4-pro --max_tokens 2000 --max_turns 20 --no_image
```

Run those commands in four terminals if you want them to execute in parallel.

## Saved Outputs

- Dialogue outputs: `results/deepseek_first100_pro_flash`
- Assessment outputs: `results/assessment/deepseek_first100_pro_flash`
- Extracted diagnoses: `results/assessment/diags/deepseek_first100_pro_flash`

## Original Reference

3MDBench paper:

```bibtex
@misc{sviridov20253mdbenchmedicalmultimodalmultiagent,
      title={3MDBench: Medical Multimodal Multi-agent Dialogue Benchmark},
      author={Ivan Sviridov and Amina Miftakhova and Artemiy Tereshchenko and Galina Zubkova and Pavel Blinov and Andrey Savchenko},
      year={2025},
      eprint={2504.13861},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2504.13861},
}
```
