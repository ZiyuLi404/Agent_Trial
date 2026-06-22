# 重构施工图（REFACTOR_PLAN）

> 本文件是项目重构的**目标态蓝图 + 动作清单**。

## ✅ 执行进度（已完成）

按方案 A（逐模块搬+改+验证）已落地：

- ✅ 8 个模块全部归位，每个目录一份 `README.md`；`AgentClinic` / `trial` / `deployment_replay` / `generate_diagnosis_distribution` 做成包，`python -m 模块.脚本` 与 `python 模块/脚本.py` 两种运行方式都通。
- ✅ 跨模块 import 全部改对并冒烟验证（`trial.*` / `deployment_replay.*` / `AgentClinic.*`）。
- ✅ 改名：`performance_estimation→history_borrowing`、`result_categorize→kl_js_divergence`、脚本同步改名。
- ✅ ICD 字典去重（只留 `AgentClinic/icd10cm_2026.jsonl`）。
- ✅ **G 三脚本合并**为 `embedding_similarity.py`（`--text_field` 必填、自动识别模型组、改读目录）；新旧 loader 在 full_dialogue/diagnosis_text 上逐字一致已验证。
- ✅ **Data*.zip 退役**：解压进 `results/generate_diagnosis_distribution/`，原 zip 移 `generated_outputs/` 留底。
- ✅ **results/ 迁移**：各模块产物归入 `results/<模块>/`。
- ✅ **两层 .gitignore**：重数据（json/jsonl/npz/pdf/transcripts）忽略，结论 CSV 入库；存量重文件已 `git rm --cached`。

**仍未做（按用户决定保留）**：anchor 复用引擎、消复制分支（§6）——会改变 anchor 数值行为，暂不动。

> 下面的目录树与动作清单是当时的**目标态蓝图**，现已基本达成，保留作参照。

---

## 0. 安全须知（务必遵守）

- **`.env` 含真实 API key，绝不提交 / 推送 GitHub。** 已确认：`.gitignore` 忽略它、版本库未跟踪、历史从未提交。
- `.env.example`（空占位模板）正常入库，供协作者复制。

---

## 1. 模块分解（8 个活模块 + 2 个归档）

分界原则：**谁真的跑问诊（用引擎生成数据） vs 谁只是事后算账（分析已有数据）。**

| 模块 | 角色 | 一句话 | 依赖 |
|------|------|--------|------|
| `AgentClinic/` | ① 引擎（地基，保留原名） | 模拟诊所：AI 医生/病人/化验师对话看病出诊断 | — |
| `trial/` | ② 在线试验框架 | 把医生当可升级版本，逐病例喂入、随机分组、写日志，测版本升级的准确率变化 | A |
| `anchor_compare/` | C 在线实验 | 新旧两版医生看同一批标准病人，不看对错、只看诊断思路像不像 | A 的拷贝 |
| `deployment_replay/` | E 在线实验 | 老病人叫不回来，用"重放病历"代替"重新问诊"估新版表现，损失多少 | A + B |
| `generate_diagnosis_distribution/` | F 在线实验（上游生产者） | 固定病例反复跑，统计诊断概率分布；可扫 模型×prompt | A |
| `embedding_similarity/` | G 离线分析 · 漂移 Method B | 把诊断/对话 embedding，算 cosine 相似度 | 吃 F 的产出 |
| `kl_js_divergence/` | I 离线分析 · 漂移 Method A | 诊断归到 ICD 病码，算分布的 KL/JS 散度 | 吃 F 的产出 |
| `history_borrowing/` | H 离线分析 | 样本太少时，从"长得像的模型"借数据修正准确率估计 | 吃 G 的相似度矩阵 |
| `figures_and_reports/` | 归档 | 画图 / PPT 脚本 | — |
| `generated_outputs/` | 归档 | png / pptx / 旧结果 zip | — |

**关键关系**
- G 和 I 是**并列的两套方法**，回答同一问题"诊断漂移多大"：G=向量+相似度，I=分布+散度。
- 数据接力：`F → G/I`（诊断数据）、`G → H`（相似度矩阵）。
- C / E 是 5 月底搭建、之后搁置的早期分支（属 design 的 borrow 主线），**先保留、标注"搁置"**，不删。

---

## 2. 目标目录树

```
Agent_Trial/
├── README.md
├── requirements.txt
├── .env / .env.example          # .env 永不入库
├── .gitignore                   # 两层 results 规则（见 §4）
│
├── AgentClinic/                 ① 引擎
│   ├── README.md
│   ├── agentclinic.py
│   ├── doctor_prompts.json      #   prompt 库（7 人设）
│   ├── agentclinic_medqa.jsonl  #   数据集 ×4
│   ├── agentclinic_*.jsonl
│   ├── icd10cm_2026.jsonl       #   ★ ICD 字典唯一来源（引擎 + I 共用）
│   └── icd10cm_codes/
│
├── trial/                       ② 在线试验框架
│   ├── README.md
│   ├── run_trial.py
│   ├── trial_manager.py
│   ├── version_manager.py
│   ├── randomization.py
│   ├── logger.py
│   └── current_version.json
│
├── anchor_compare/              C
│   ├── README.md
│   └── anchor_compare.py        #   ※待办：复用引擎、消复制分支
│
├── deployment_replay/           E
│   ├── README.md
│   ├── deployment_timeline.py
│   ├── replay_evaluator.py
│   ├── oracle_evaluator.py
│   └── hybrid_estimator.py
│
├── generate_diagnosis_distribution/   F
│   ├── README.md
│   ├── diagnosis_distribution.py   #   ★ 统一 runner，--doctor_llm 传任意模型
│   └── make_dist_report.py         #   配套报告（从归档收回）
│                                   #   ✗ run_gpt_data.sh 删除
│
├── embedding_similarity/        G
│   ├── README.md
│   ├── embedding_similarity.py     #   ★ 三脚本合一，参数化
│   └── requirements.txt            #   ← full_dialogue_embedding_requirements.txt
│
├── kl_js_divergence/            I
│   ├── README.md
│   └── icd_categorize_compare.py   #   ✗ 本地 ICD 字典删除，引用引擎的
│
├── history_borrowing/           H（5 个=5 个真步骤，保持拆分）
│   ├── README.md
│   ├── accuracy_summary.py
│   ├── history_borrowing.py
│   ├── run_all_orders.py
│   ├── train_borrow_params.py
│   └── visualize_borrow_params.py
│
├── results/                     ★ 公共货架（两层 gitignore）
│   ├── trial/                              trial_log.jsonl            [ignore]
│   ├── deployment_replay/                  deployment_log.jsonl, transcripts/ [ignore]
│   ├── generate_diagnosis_distribution/    ← 原 Data*.zip 解成普通文件夹
│   │   ├── deepseek_flash_1/case_X.json   [ignore]
│   │   ├── gpt_5_5_1/...                   #   G 和 I 都从这里读
│   │   └── ...
│   ├── embedding_similarity/               *.npz [ignore] · 相似度矩阵 CSV [keep]
│   ├── kl_js_divergence/                   散度 / summary CSV [keep]
│   │   ├── deepseek_flash_vs_pro/
│   │   ├── deepseek_vs_gpt/
│   │   └── prompt_compare/
│   └── history_borrowing/                  accuracy_by_25_cases.csv 等
│
├── figures_and_reports/         归档（画图/PPT 脚本）
├── generated_outputs/           归档（png/pptx/旧结果 zip）
└── notes/                       个人笔记（不动）
```

---

## 3. 核心设计决定

### 3.1 F / G 各合并成 1 个参数化脚本（消灭"按数据组复制"）
- **G**：三个 `case_level_embedding_*.py`（各 ~600 行，95% 重复）→ 1 个 `embedding_similarity.py`。
  - 模型组**从文件夹名自动推断**（`deepseek_flash_1` → 去尾部 `_1` → 组名 `deepseek_flash`）；删掉 `MODEL_ORDER` / `detect_model_from_group` 硬编码。
  - `--text_field` **必填、无默认、限定 choices**（`diagnosis_text` | `full_dialogue`），两个实验并列对等：
    - `diagnosis_text`：只比最终诊断（结论层面）
    - `full_dialogue`：比整段问诊（过程层面）
  - `--data_dir` 指向 `results/generate_diagnosis_distribution/`。
- **F**：`diagnosis_distribution.py` 已支持 `--doctor_llm` 扫多模型；GPT 只是另一个模型值 → **删 `run_gpt_data.sh`**（它唯一独有的"手工 zip"步骤随 zip 一起退役）。

### 3.2 `Data*.zip` 退役（修最大的设计债）
- 真相：这三个 zip 是 **F 的输出，被同学在 Mac 上手工打包**（含 `__MACOSX/`、`.DS_Store` 垃圾），G 再用 `zipfile` 直接读。
- 改造：F 本来就写普通文件夹 → **不再打包**，落到 `results/generate_diagnosis_distribution/<model>/case_X.json`；G 改**读目录**（删 `zipfile` + `__MACOSX/._` 过滤逻辑）。
- 旧 `Data*.zip` 解开后即可弃用（或丢 `generated_outputs/` 留底）。

### 3.3 ICD 字典去重
- 只保留 `AgentClinic/icd10cm_2026.jsonl` 一份；`kl_js_divergence/` 删除本地副本，改引用引擎路径。

### 3.4 代码 / 数据分离
- 模块目录 = 代码 + README；所有"跑出来的数据"进 `results/<模块>/`。
- 例外：`current_version.json`、`doctor_prompts.json` 属代码/配置，留模块内。

---

## 4. 两层 .gitignore 规则

> 大块可再生数据忽略；小块最终结论入库。

```gitignore
# 重的、可再跑出来的 → 忽略
results/**/case_*.json
results/**/transcripts/
results/**/*.npz
results/**/*.jsonl
results/**/*.pdf
# 小而宝贵的最终结论（相似度矩阵 / 散度 / summary CSV）→ 默认入库
```

注意：现有 `.gitignore` 用了激进的 `*.json` / `*.csv` / `results/` 全局忽略，大改时需替换为上面的细粒度规则（否则会误伤 `doctor_prompts.json` 等代码内 json，且无法入库结论 CSV）。

---

## 5. 执行动作清单（最后统一改时照做）

1. 散在根目录的脚本 → 各归其模块文件夹（用 `git mv` 保留历史）。
2. trial 五件套 → `trial/`；随之修所有 `from trial_manager import ...` 等 import。
3. F/G 各合并成 1 个参数化脚本；删 `run_gpt_data.sh`、三个 `case_level_*.py`。
4. `Data*.zip` 退役 → 解成目录；G 改读目录、删 zip 过滤逻辑。
5. ICD 字典去重：I 引用引擎那份。
6. `make_dist_report.py` 从归档收回 F。
7. 产物路径全部改成 `results/<模块>/`（修脚本里的裸相对路径）。
8. 替换 `.gitignore` 为两层规则。
9. 每模块补一份 `README.md`。
10. 改名：`result_categorize → kl_js_divergence`、`performance_estimation → history_borrowing`、`diagnosis_distribution.py` 所在目录、`anchor` 文件等。
11. C / E 保留并在各自 README 标注"早期搭建、当前搁置"。

---

## 6. 已知技术债（重构时一并处理）

- **`anchor_compare` 是引擎的整段复制分支**（~600 行），且其 `query_model` 停留在旧版（`temperature=0`、`max_tokens=200`，易截断结构化输出）。理想改法：改成 `from AgentClinic.agentclinic import ...`，仅以薄子类保留 `output_format="anchor_compare"` 的结构化输出差异。**当前用户决定先不动。**
- `randomization._block` 不跨进程持久化。
