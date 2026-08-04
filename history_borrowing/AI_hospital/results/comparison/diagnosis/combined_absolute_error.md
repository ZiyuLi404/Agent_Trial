# Combined absolute-error comparison

| Model | Algorithm 1 bundle | Algorithm 1 borrowed | Algorithm 2 bundle | Algorithm 2 borrowed |
|---|---:|---:|---:|---:|
| gpt3 | 0.0800 | 0.0333 | 0.0577 | 0.0577 |
| gpt4 | 0.0400 | 0.0467 | 0.0229 | 0.0305 |
| qwen_max | 0.0500 | 0.0567 | 0.0327 | 0.0380 |
| wenxin | 0.0600 | 0.0267 | 0.0730 | 0.0659 |
| Mean (MAE) | 0.0575 | 0.0409 | 0.0466 | 0.0480 |

Algorithm 1 errors use full-data observed accuracy as the reference. Algorithm 2 errors use the full-data Beta posterior mean as the reference.
