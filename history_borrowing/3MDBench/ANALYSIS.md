# 3MDBench 80-case analysis

## Coverage

The updated export contains 80 shared cases for four models, or 320
model/case assessments. Version bundles contain 20 cases each.

| Model | Passing competence | Accuracy | Missing competence |
|---|---:|---:|---:|
| deepseek_flash | 63/80 | 0.7875 | 15 |
| deepseek_pro | 57/80 | 0.7125 | 20 |
| gpt5_mini | 45/80 | 0.5625 | 34 |
| qwen_plus | 46/80 | 0.5750 | 31 |

Passing means rubric item 4.1 is `satisfactory` or `excellent`. Missing or
truncated assessments are conservatively failures, so differences in missing
rates materially affect these observed accuracies.

## Assessment-profile embedding similarity

| Model | deepseek_flash | deepseek_pro | gpt5_mini | qwen_plus |
|---|---:|---:|---:|---:|
| deepseek_flash | 1.000 | 0.940 | 0.919 | 0.923 |
| deepseek_pro | 0.940 | 1.000 | 0.932 | 0.912 |
| gpt5_mini | 0.919 | 0.932 | 1.000 | 0.902 |
| qwen_plus | 0.923 | 0.912 | 0.902 | 1.000 |

These values compare canonical evaluator profiles, not dialogue content.

## History borrowing

Algorithm 1 uses one 20-case version bundle per model. Assessment similarity
reduces MAE from 0.0969 to 0.0219 with fitted `alpha=0.6` and `lambda=0`.
Because lambda is zero, its best in-sample fit borrows uniformly rather than
using differences in assessment similarity.

For the canonical Algorithm 2 order
`deepseek_flash -> deepseek_pro -> gpt5_mini -> qwen_plus`, bundle posterior
mean MAE is 0.0820 and history-borrowed posterior mean MAE is 0.0713. Algorithm
2 also evaluates all 24 deployment orders; the lowest-MAE order is
`deepseek_pro -> qwen_plus -> deepseek_flash -> gpt5_mini` with mean posterior
MAE 0.0273.

## Limitations

- Missing assessments are treated as failures to retain all 80 aligned cases.
- The embedding and outcome are derived from the same evaluator assessment,
  which can induce circularity.
- Algorithm 1 hyperparameters are fitted against the same full-data outcomes
  used for reporting, so its improvement is in-sample.
- The source does not permit diagnosis- or conversation-level similarity. Those
  old 50-case matrices were removed rather than silently reused.
