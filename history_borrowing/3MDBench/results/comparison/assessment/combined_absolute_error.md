# Combined absolute-error comparison

| Model | Algorithm 1 bundle | Algorithm 1 borrowed | Algorithm 2 bundle | Algorithm 2 borrowed |
|---|---:|---:|---:|---:|
| deepseek_flash | 0.0625 | 0.0375 | 0.0377 | 0.0377 |
| deepseek_pro | 0.1375 | 0.0375 | 0.1109 | 0.1215 |
| gpt5_mini | 0.1125 | 0.0008 | 0.1064 | 0.0767 |
| qwen_plus | 0.0750 | 0.0117 | 0.0732 | 0.0494 |
| Mean (MAE) | 0.0969 | 0.0219 | 0.0820 | 0.0713 |

Algorithm 1 errors use full-data observed accuracy as the reference. Algorithm 2 errors use the full-data Beta posterior mean as the reference.
