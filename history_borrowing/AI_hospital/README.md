# AI Hospital history-borrowing data

This directory adapts the first 100 shared cases from
`results/AI_Hospital_fingerprint` to the formats used by the two shared
history-borrowing algorithms. It includes four doctor models: `gpt3`, `gpt4`,
`qwen_max`, and `wenxin`.

## Case and outcome definition

“First 100” means the first 100 numerically sorted scenario IDs in the
intersection of all four model directories. The IDs are sparse and run from
1001 through 1205; the exact list is in `dataset_summary.json`.

The local fingerprint files have an empty `correct_diagnosis_reference`.
References and grades therefore come from the official AI Hospital repository:

- expert diagnoses: `src/data/patients.json`
- per-case GPT-4 evaluation: `src/outputs/evaluation/evaluation_iiyi_gpt4_5part.jsonl`

The official evaluator assigns diagnosis grades A–D. For the binary outcome
required by the shared history-borrowing code, A/B map to 1 and C/D map to 0.
This follows the evaluator rubric: A is correct, B is basically correct, C
contains diagnostic errors, and D is incorrect.

The published WenXin evaluation for case 1201 is truncated before the diagnosis
grade. Its partial/mismatched diagnosis is conservatively imputed as C. This is
the only imputation and is marked in every relevant audit artifact.

## Contents

- `groundtruth/*.json` — 100 binary diagnosis outcomes per model.
- `embedding_inputs/<model>/case_*.json` — extracted diagnosis conclusions and
  doctor-only dialogue text.
- `official_reference_first_100.json` — expert diagnoses for the selected IDs.
- `official_diagnosis_grades_first_100.csv` — official grades and their binary
  mapping.
- `accuracy_by_25_cases.csv` — four 25-case buckets for the four-model
  borrowing experiment.
- `accuracy_by_10_cases.csv` — descriptive ten-case buckets.
- `extraction_audit.csv` — source, extraction, grade, and output audit.
- `similarity_matrix/{diagnosis,conversation}/` — case-level similarity,
  aggregate matrices, raw text, normalized embeddings, and metadata.
- `results/algorithm_1/` and `results/algorithm_2/` — completed borrowing runs.
- `ANALYSIS.md` — concise findings and limitations.

## Rebuild the prepared inputs

Download the two official files:

```bash
curl -L --fail \
  https://raw.githubusercontent.com/LibertFan/AI_Hospital/main/src/data/patients.json \
  -o /private/tmp/ai_hospital_patients.json

curl -L --fail \
  https://raw.githubusercontent.com/LibertFan/AI_Hospital/main/src/outputs/evaluation/evaluation_iiyi_gpt4_5part.jsonl \
  -o /private/tmp/evaluation_iiyi_gpt4_5part.jsonl
```

Then prepare the data:

```bash
python3 history_borrowing/AI_hospital/prepare_data.py
```

The preparation script verifies that each local diagnosis output is byte-for-
byte identical, after trimming, to the output evaluated by the official file.

## Rebuild embedding similarity

Diagnosis-conclusion similarity:

```bash
python history_borrowing/AI_hospital/compute_similarity.py \
  --text_field diagnosis_text \
  --model Qwen/Qwen3-Embedding-0.6B \
  --max_seq_length 256 \
  --output_dir history_borrowing/AI_hospital/similarity_matrix/diagnosis
```

Doctor-only conversation similarity:

```bash
python history_borrowing/AI_hospital/compute_similarity.py \
  --text_field doctor_dialogue \
  --model Qwen/Qwen3-Embedding-0.6B \
  --max_seq_length 256 \
  --output_dir history_borrowing/AI_hospital/similarity_matrix/conversation
```

The aggregation is the mean of cosine similarities between two models on the
same patient. Expert diagnoses are never embedded as model outputs.

## Run Algorithm 1

Use either `diagnosis` or `conversation` for `<source>`:

```bash
python3 history_borrowing/algorithm_1/history_borrowing.py \
  --accuracy_csv history_borrowing/AI_hospital/accuracy_by_version_bundle.csv \
  --similarity_csv history_borrowing/AI_hospital/similarity_matrix/<source>/mean_model_similarity_matrix.csv \
  --replicate_map_json history_borrowing/AI_hospital/replicate_map.json \
  --output_csv history_borrowing/AI_hospital/results/algorithm_1/<source>/results.csv \
  --output_summary_json history_borrowing/AI_hospital/results/algorithm_1/<source>/summary.json
```

Generate the Algorithm 1 numeric comparison and the combined Algorithm 1/2
absolute-error table:

```bash
python3 history_borrowing/algorithm_1/numeric_comparison.py \
  --algorithm1_results history_borrowing/AI_hospital/results/algorithm_1/<source>/results.csv \
  --algorithm2_comparison history_borrowing/AI_hospital/results/algorithm_2/<source>/distribution_comparison.csv \
  --output_dir history_borrowing/AI_hospital/results/comparison/<source> \
  --dataset_name "AI Hospital" \
  --similarity_name <source>
```

## Run Algorithm 2

```bash
python3 history_borrowing/algorithm_2/bayesian_pseudo_posterior.py \
  --data_dir history_borrowing/AI_hospital/groundtruth \
  --similarity_file history_borrowing/AI_hospital/similarity_matrix/<source>/mean_model_similarity_matrix.csv \
  --lambda 10 \
  --alpha0 1 \
  --beta0 1 \
  --credible_level 0.95 \
  --all_orders \
  --output_dir history_borrowing/AI_hospital/results/algorithm_2/<source>
```

Generate the version-bundle, history-borrowed, and full 100-case posterior
distribution comparison. AI Hospital has 100 cases and four model versions, so
each version bundle contains 25 cases:

```bash
python3 history_borrowing/algorithm_2/distribution_comparison.py \
  --data_dir history_borrowing/AI_hospital/groundtruth \
  --similarity_file history_borrowing/AI_hospital/similarity_matrix/<source>/mean_model_similarity_matrix.csv \
  --output_dir history_borrowing/AI_hospital/results/algorithm_2/<source> \
  --dataset_name "AI Hospital" \
  --fresh_strategy version_bundle
```

The output includes a numeric CSV, Markdown table, overlaid Beta-density plot,
and a 95% credible-interval table image.

## Interpretation

These are exploratory, in-sample borrowing results. Algorithm 1 selects
hyperparameters against the same 100-case totals used to report improvement,
and model similarity is calculated on the same cases used for outcomes. A
confirmatory experiment should fit borrowing parameters and similarity on a
separate calibration set, then evaluate on unseen cases.
