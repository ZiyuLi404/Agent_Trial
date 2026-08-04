# 3MDBench 80-case history-borrowing data

This package adapts the four-model, 80-case assessment export in
`results/3MDBench` to the shared history-borrowing algorithms.

## Source schema and outcome

The updated source contains rubric assessments only; it does not contain the
doctor dialogues, predicted diagnoses, or reference diagnoses used by the old
50-case package. The four source folders map to:

- `deepseek_flash_doctor` → `deepseek_flash`
- `deepseek_pro_doctor` → `deepseek_pro`
- `gpt5_mini_doctor` → `gpt5_mini`
- `qwen_plus_doctor` → `qwen_plus`

All four models share 80 sparse case IDs. The binary outcome is derived from
rubric item 4.1, Overall Clinical Competence:

- `satisfactory` or `excellent` → 1
- `unsatisfactory`, missing, or truncated before item 4.1 → 0

This conservative missing-as-failure decision is recorded in
`assessment_audit.csv` and `dataset_summary.json`.

## Similarity and version bundles

Each assessment is canonicalized into assessment status, seven binary rubric
items, and the competence label. Qwen3-Embedding-0.6B embeds those canonical
profiles. The resulting matrix measures evaluator-profile similarity; it is
not dialogue or diagnosis semantic similarity.

The 80 ordered cases are divided evenly across four model versions, producing
four contiguous 20-case bundles.

## Rebuild

```bash
python3 history_borrowing/3MDBench/prepare_data.py

python history_borrowing/3MDBench/compute_similarity.py \
  --text_field assessment_text \
  --model Qwen/Qwen3-Embedding-0.6B \
  --max_seq_length 128 \
  --output_dir history_borrowing/3MDBench/similarity_matrix/assessment

python3 history_borrowing/algorithm_1/version_bundle_accuracy.py \
  --groundtruth_dir history_borrowing/3MDBench/groundtruth \
  --output_csv history_borrowing/3MDBench/accuracy_by_version_bundle.csv
```

## Algorithm 1

```bash
python3 history_borrowing/algorithm_1/history_borrowing.py \
  --accuracy_csv history_borrowing/3MDBench/accuracy_by_version_bundle.csv \
  --similarity_csv history_borrowing/3MDBench/similarity_matrix/assessment/mean_model_similarity_matrix.csv \
  --replicate_map_json history_borrowing/3MDBench/replicate_map.json \
  --output_csv history_borrowing/3MDBench/results/algorithm_1/assessment/results.csv \
  --output_summary_json history_borrowing/3MDBench/results/algorithm_1/assessment/summary.json
```

## Algorithm 2

```bash
python3 history_borrowing/algorithm_2/bayesian_pseudo_posterior.py \
  --data_dir history_borrowing/3MDBench/groundtruth \
  --similarity_file history_borrowing/3MDBench/similarity_matrix/assessment/mean_model_similarity_matrix.csv \
  --lambda 10 --alpha0 1 --beta0 1 --credible_level 0.95 --all_orders \
  --output_dir history_borrowing/3MDBench/results/algorithm_2/assessment

python3 history_borrowing/algorithm_2/full_posterior.py \
  --data_dir history_borrowing/3MDBench/groundtruth \
  --expected_n 80 \
  --output_csv history_borrowing/3MDBench/results/algorithm_2/full_posterior.csv
```

## Comparison tables

```bash
python3 history_borrowing/algorithm_2/distribution_comparison.py \
  --data_dir history_borrowing/3MDBench/groundtruth \
  --similarity_file history_borrowing/3MDBench/similarity_matrix/assessment/mean_model_similarity_matrix.csv \
  --output_dir history_borrowing/3MDBench/results/algorithm_2/assessment \
  --dataset_name 3MDBench --fresh_strategy version_bundle

python3 history_borrowing/algorithm_1/numeric_comparison.py \
  --algorithm1_results history_borrowing/3MDBench/results/algorithm_1/assessment/results.csv \
  --algorithm2_comparison history_borrowing/3MDBench/results/algorithm_2/assessment/distribution_comparison.csv \
  --output_dir history_borrowing/3MDBench/results/comparison/assessment \
  --dataset_name 3MDBench --similarity_name assessment
```

See `ANALYSIS.md` for results and limitations.
