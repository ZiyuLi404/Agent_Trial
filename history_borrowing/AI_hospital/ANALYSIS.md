# AI Hospital first-100 analysis

## Coverage and scoring

All four model directories share 505 scenario IDs. This package selects the
first 100 sorted IDs, from 1001 to 1205, and has exactly one output per
model/case (400 total).

The source fingerprint field `correct_diagnosis_reference` is empty in every
selected file. The package fills references from the official 506-record
patient dataset and outcomes from the official GPT-4 five-part evaluation.
Every local diagnosis output matches the corresponding officially evaluated
output.

Diagnosis grades are binarized as A/B correct and C/D incorrect:

| Model | A | B | C | D | Correct | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| gpt3 | 2 | 18 | 33 | 47 | 20/100 | 0.20 |
| gpt4 | 2 | 22 | 31 | 45 | 24/100 | 0.24 |
| qwen_max | 1 | 24 | 24 | 51 | 25/100 | 0.25 |
| wenxin | 1 | 17 | 17 | 65 | 18/100 | 0.18 |

WenXin case 1201 accounts for the only non-official grade: the published
evaluation stops during its diagnosis analysis, so the partial/mismatched
output is conservatively imputed as C.

## Embedding similarity

The matrices use Qwen3-Embedding-0.6B, normalized embeddings, a 256-token cap,
and same-case cosine similarity averaged across 100 patients.

Diagnosis conclusions:

| Model | gpt3 | gpt4 | qwen_max | wenxin |
|---|---:|---:|---:|---:|
| gpt3 | 1.000 | 0.756 | 0.695 | 0.654 |
| gpt4 | 0.756 | 1.000 | 0.728 | 0.644 |
| qwen_max | 0.695 | 0.728 | 1.000 | 0.676 |
| wenxin | 0.654 | 0.644 | 0.676 | 1.000 |

Doctor-only conversations:

| Model | gpt3 | gpt4 | qwen_max | wenxin |
|---|---:|---:|---:|---:|
| gpt3 | 1.000 | 0.717 | 0.709 | 0.766 |
| gpt4 | 0.717 | 1.000 | 0.741 | 0.705 |
| qwen_max | 0.709 | 0.741 | 1.000 | 0.724 |
| wenxin | 0.766 | 0.705 | 0.724 | 1.000 |

The similarity structures differ materially. GPT-3/GPT-4 are the closest pair
by diagnosis conclusion, while GPT-3/WenXin are the closest pair by
consultation behavior. This supports keeping the two similarity sources
separate rather than treating them as interchangeable.

## History borrowing

Algorithm 1 uses one distinct 25-case bucket per model and compares the
borrowed estimate with each model's full 100-case accuracy.

| Similarity | Baseline MAE | Borrowed MAE | Relative change | Best alpha | Best lambda |
|---|---:|---:|---:|---:|---:|
| Diagnosis | 0.0575 | 0.0408 | 29.0% lower | 0.5 | 0 |
| Conversation | 0.0575 | 0.0274 | 52.3% lower | 0.5 | 200 |

The diagnosis result selects lambda 0, meaning its best in-sample fit ignores
diagnosis-distance differences and borrows uniformly. Conversation similarity
selects a high lambda and produces the lower in-sample MAE.

Algorithm 2 was run over all 24 deployment orders with lambda 10 and a
Beta(1,1) prior. For both similarity sources, the lowest-MAE order is:

`wenxin -> gpt4 -> qwen_max -> gpt3`

Its mean posterior MAE is 0.01848 with diagnosis similarity and 0.01847 with
conversation similarity. Across all orders, conversation similarity supplies
more effective borrowed prior observations on average, while model-level error
patterns remain similar.

## Limitations

- Algorithm 1's hyperparameters are selected and assessed on the same 100
  cases, so the reported MAE reductions are optimistic.
- Similarity and accuracy use the same patients. A confirmatory study should
  calculate similarity on a disjoint calibration set.
- A/B versus C/D is a documented binarization of an ordinal evaluator rubric,
  not an exact-match disease-entity score.
- One of 400 grades is transparently imputed.
- Conversation embeddings are capped at 256 tokens, so later turns may be
  truncated.
