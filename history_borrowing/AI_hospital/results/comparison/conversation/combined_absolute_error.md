# Combined absolute-error comparison

| Model | Algorithm 1 bundle | Algorithm 1 borrowed | Algorithm 2 bundle | Algorithm 2 borrowed |
|---|---:|---:|---:|---:|
| gpt3 | 0.0800 | 0.0200 | 0.0577 | 0.0577 |
| gpt4 | 0.0400 | 0.0403 | 0.0229 | 0.0282 |
| qwen_max | 0.0500 | 0.0494 | 0.0327 | 0.0387 |
| wenxin | 0.0600 | 0.0000 | 0.0730 | 0.0573 |
| Mean (MAE) | 0.0575 | 0.0274 | 0.0466 | 0.0455 |

Algorithm 1 errors use full-data observed accuracy as the reference. Algorithm 2 errors use the full-data Beta posterior mean as the reference.
