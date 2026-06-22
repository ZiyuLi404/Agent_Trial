# kl_js_divergence — 诊断漂移 · Method A（分布 + 散度）（I）

## 这个模块干嘛
检测**模型 / prompt 升级带来的诊断漂移**。和 **G `embedding_similarity`（Method B，向量+相似度）是并列的两套方法**——本模块走"符号/编码"路线：

1. **归类**：把五花八门的诊断**自由文字**（`"pneumonia"` / `"lung infection"` / `"肺炎"`…）用 LLM 映射到**标准 ICD-10-CM 病码**，统一口径。
2. **比较**：在 ICD 码层面重算分布，再两两算 **KL 散度 / JS 散度**，量两批结果差多少。

属**离线分析**，不跑问诊、不碰引擎（自带 OpenAI 客户端）。是一条**到头的分析支线**——产物直接给研究者看 / 进图 / 进论文，不喂任何下游模块。

## 输入 / 输出
- 输入：结果文件夹里的 `diagnosis_text`（来自 **F `generate_diagnosis_distribution`** 等跑出的数据）
- 输出：
  - `summary.icd10.csv` —— 每个文件夹在 ICD 码层面的分布
  - `folder_similarity_matrix_js.csv` / `folder_similarity_matrix_symmetric_kl.csv` —— 文件夹两两散度
  - `pairwise_case_metrics.csv` —— 逐 case 指标

## 三种现成用法（看子目录）
| 子目录 | 在比 |
|--------|------|
| `deepseek_flash_vs_pro/` | 同家族不同档位模型 |
| `deepseek_vs_gpt/` | 跨厂商模型 |
| `prompt_compare/` | 不同 prompt 人设（**prompt 实验**那条线） |

## 怎么跑
```bash
# 从仓库根目录运行
# 归类一个文件夹：
python kl_js_divergence/icd_categorize_compare.py \
    --mode categorize --folders 50case_10runs_flash \
    --cases 0-19 --runs 0-9 --temperature temp_0.05 --run_name analysis_v1
# 归类+比较一起做：--mode both
```

## 关键依赖
- ICD 字典：**引用引擎唯一来源** `AgentClinic/icd10cm_2026.jsonl`（已去重，本模块不再保留副本）。
- LLM：需 `OPENAI_API_KEY`（用于把诊断文字映射到 ICD 码）。

## 重构记录
- ✅ 改名 `result_categorize → kl_js_divergence`；脚本 `icd10_categorize_compare.py → icd_categorize_compare.py`。
- ✅ ICD 字典去重，默认 `--icd_dict` 指向 `AgentClinic/icd10cm_2026.jsonl`。
- ✅ 比较结果数据（`deepseek_flash_vs_pro` / `deepseek_vs_gpt` / `prompt_compare`）已迁到 `results/kl_js_divergence/`；`--out_dir` / `--cache_file` 默认值同步指向那里。本目录现只剩代码 + README。
