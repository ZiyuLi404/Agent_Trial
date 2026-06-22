# embedding_similarity — 诊断漂移 · Method B（向量 + 相似度）（G）

## 这个模块干嘛
检测**模型 / prompt 升级带来的诊断漂移**。和 **I `kl_js_divergence`（Method A，分布+散度）是并列的两套方法**——本模块走"向量"路线：把诊断/对话文本 **embedding 成向量**，按 case 取平均、归一化，再算 **cosine 相似度**，比模型/组之间有多像。

属**离线分析**，不跑问诊、不碰引擎（只用 sentence-transformers / numpy / pandas / sklearn）。输入是 **F `generate_diagnosis_distribution`** 跑出的 `case_X.json`。

## 两个并列实验，由 `--text_field` 决定（必填、无默认）
| `--text_field` | 嵌入什么 | 回答 |
|----------------|---------|------|
| `diagnosis_text` | 最终诊断那一句 | 两模型**结论**像不像 |
| `full_dialogue` | 整段问诊对话 | 两模型**过程**像不像 |

## 怎么跑
```bash
# 从仓库根目录。数据目录下形如 <group>/case_*.json
python embedding_similarity/embedding_similarity.py \
    --data_dir results/generate_diagnosis_distribution \
    --text_field diagnosis_text \
    --model Qwen/Qwen3-Embedding-0.6B
# 输出默认落到 results/embedding_similarity/<text_field>/
```
模型组**从文件夹名自动推断**（去掉尾部批次号 `_<run>`）：`deepseek_flash_1/2 → deepseek_flash`、`gpt_5_5_1/2 → gpt_5_5`。**加新模型无需改代码**。

## 输出
`mean_group_similarity_matrix.csv`、`mean_model_similarity_matrix.csv`、`case_level_*`、`*.npz` 等 → 其中相似度矩阵会喂给 **H `history_borrowing`**。

## 重构记录
- ✅ **三脚本合并成一个** `embedding_similarity.py`：`full_dialogue` / `analyze_data2` / `gpt` 的唯一差别（嵌入哪段文本）变成 `--text_field` 参数。
- ✅ **改读目录**（`--data_dir`），不再读 `Data*.zip`，删掉 `zipfile` + `__MACOSX/._` 过滤逻辑。
- ✅ **消灭硬编码**：删 `MODEL_ORDER` / 关键词版 `detect_model_from_group`，改为按文件夹名自动推断模型组。
- ✅ **行为验证**：新旧 loader 在 `full_dialogue`（900 行）和 `diagnosis_text`（gpt 360 行）上 `(group,case,run,文本)` 逐字一致；相似度数学为原样复制。model 级标签由 `flash/qwen` 变为更具描述性的 `deepseek_flash/Qwen_plus_turbo`（分组与数值不变）。
- 旧 `Data*.zip` 已退役到 `generated_outputs/`（留底）；解压后的数据在 `results/generate_diagnosis_distribution/`。
- 旧结果目录 `embedding_full_dialogue_results*` 已迁到 `results/embedding_similarity/`。
