# `diagnosis_distribution.py` 使用说明

> 更新日期：2026-05-31

> 一句话：**固定一个病例，让 AI 医生反复问诊很多次，统计它每次诊断结果，得到这个病例上诊断结果的概率分布。**

这个脚本是在 AgentClinic 基础上做的一个独立实验，和仓库里另一个 `agentclinic_anchor_compare.py`（锚定回归实验）**互不相关**，不要混淆。

---

## 1. 这个实验在干什么

AgentClinic 里，一个病例（case）会让两个 LLM 扮演 **doctor（医生）** 和 **patient（病人）** 反复对话问诊，最后医生给出一个诊断。

我们发现：**同一个病例、同一个模型，每次跑出来的诊断可能不一样**（采样随机性 + 服务端非确定性）。

于是这个脚本做的事就是用**大数定律**的思路：
- 把同一个病例**从头到尾完整重跑 N 次**（每次都是一整段独立的问诊对话）；
- 收集医生 N 次的最终诊断；
- 按结果分桶计数 → 得到「每种诊断出现的频率」≈ 真实概率分布。

输出既有**统计分布**（每种诊断占百分之几、有多发散），也有**逐次完整对话原始记录**（方便以后任意方式再分析）。

---

## 2. 准备工作

### 依赖
```bash
pip install -r requirements.txt        # 主要是 openai、python-dotenv
```

### API Key
脚本启动会自动读 `.env`（已 gitignore，不会上传）。复制模板并填入你自己的 key：
```bash
cp .env.example .env
# 编辑 .env，至少填 DEEPSEEK_API_KEY
```
`.env` 里需要：
```
DEEPSEEK_API_KEY=sk-xxxxxxxx
```
（也可以用命令行 `--deepseek_api_key` 传，但不推荐写进命令历史。）

---

## 3. 最简单的跑法

```bash
python -u diagnosis_distribution.py --scenario_ids 0 --runs 30
```

含义：拿 MedQA 第 0 号病例，完整重跑 30 次，统计诊断分布。结果默认写到 `results/diag_dist_<时间戳>/`。

> 加 `-u` 是让输出实时刷新，可以一边跑一边 `tail -f` 看进度。

跑多个病例：
```bash
python -u diagnosis_distribution.py --scenario_ids 0-4 --runs 30 --out_dir results/run1
```
`--scenario_ids` 支持 `0,1,2` / `0-9`（区间）/ `all`（全部病例）。

---

## 4. 默认行为（这一版的关键设定）

| 设定 | 默认值 | 说明 |
|---|---|---|
| 重复粒度 | 整段问诊独立重跑 | 每个样本从头跑完整 doctor↔patient↔measurement 多轮对话 |
| doctor 模型 | `deepseek-v4-pro` | 研究对象，用强模型 |
| patient 模型 | `deepseek-v4-flash` | 病人/辅助，用快模型省时间 |
| 温度 | `0.05` | 引擎生产值；最初观察到「每次不一样」的温度 |
| 分桶方式 | `exact` | **按原始字符串**分桶，大小写/空格无关；语义相同但写法不同的**不合并** |
| 判对错 | **关** | 不判诊断对不对；金标准只作参考存进文件 |
| moderator 裁判 | **不启用** | 默认全程不调用裁判模型，只积累原始数据 |
| 并发 | `1`（串行） | 不并行 |

**重点：默认模式下只「积累原始数据」，不做任何主观判定。** 金标准（正确答案）会写进文件字段 `correct_diagnosis_reference` 供以后参考，但不参与统计。

---

## 5. 输出长什么样

### 目录结构
```
results/run1/
├── temp_0.05/
│   ├── case_0.json      # 第 0 号病例的完整结果
│   ├── case_1.json
│   └── ...
└── summary.json         # 所有病例的统计摘要
```

### 单个病例 `case_0.json`（核心字段）
```jsonc
{
  "scenario_id": 0,
  "correct_diagnosis_reference": "Myasthenia gravis",  // 正确答案，仅参考
  "runs": 30,
  "num_distinct_buckets": 3,        // 出现了几种不同诊断
  "entropy_bits": 1.371,            // 香农熵：越大越发散，0=每次都一样
  "mode_diagnosis": "Myasthenia gravis",  // 最常出现的诊断
  "mode_prob": 0.6,                 // 占 60%

  "distribution": [                 // ⭐ 结果概率分布，prob 之和=1，按频率降序
    { "canonical": "Myasthenia gravis", "count": 18, "prob": 0.60, "members": {...} },
    { "canonical": "Lambert-Eaton myasthenic syndrome", "count": 7, "prob": 0.23, ... },
    { "canonical": "Guillain-Barré syndrome", "count": 5, "prob": 0.17, ... }
  ],

  "samples": [                      // ⭐ 逐次原始记录，每次重跑一条
    {
      "run": 0,
      "diagnosis_text": "Myasthenia gravis",   // 这次的最终诊断
      "forced": false,             // 是否被「强制收尾」（见第 7 节）
      "num_infs": 12,              // 这次问了多少轮
      "elapsed_sec": 138.4,
      "bucket": "Myasthenia gravis",
      "full_dialogue": "Doctor: ...\nPatient: ...\n..."  // 完整问诊全文
    },
    ...
  ]
}
```

简单记：**`distribution` 是统计结论，`samples` 是逐次原料（含完整对话）。**

### 终端输出
跑的时候终端会实时打印每次结果，跑完打印分布表，例如：
```
┌─ Case 0  (temp=0.05, N=30) ────────
│  gold(参考): Myasthenia gravis
  run  1/30  done [ 1/30]  138.4s  infs=12 forced=N  | Myasthenia gravis
  ...
│  诊断分布 (N=30, distinct=3, entropy=1.371 bits):
│     60.0%  (18/30)  Myasthenia gravis
│     23.3%  ( 7/30)  Lambert-Eaton myasthenic syndrome
│     16.7%  ( 5/30)  Guillain-Barré syndrome
└─ saved → results/run1/temp_0.05/case_0.json
```

---

## 6. 可选功能（默认都关，需要时再开）

### 判对错（会调用裁判模型）
```bash
python -u diagnosis_distribution.py --scenario_ids 0 --runs 30 --grade_correctness
```
开启后每个诊断会和金标准比对，输出里多出 `p_correct`（正确率）和每个桶的 `is_correct` 标记。

### 语义合并（会调用裁判模型）
```bash
python -u diagnosis_distribution.py --scenario_ids 0 --runs 30 --bucketing semantic
```
把「表述不同但语义相同」的诊断（如 *Pneumonia* / *Community-acquired pneumonia*）合并成一个桶。默认的 `exact` 不会合并。

### 温度对照
```bash
python -u diagnosis_distribution.py --scenario_ids 0 --runs 30 --temperatures 0,0.05,0.7
```
同一批病例在多个温度下各跑一遍，分目录保存，末尾打印对照表。用来区分「服务端底噪（temp=0）」和「采样发散（temp>0）」。

### 并发提速
```bash
python -u diagnosis_distribution.py --scenario_ids 0 --runs 30 --concurrency 5
```
同一病例的 N 次重跑并发进行（互相独立）。能显著提速，但 `--verbose` 时多次对话会交错（每行带 `[cX rY]` 前缀可区分）。

### 完整参数表
| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `MedQA` | 数据集：MedQA / MedQA_Ext / NEJM / NEJM_Ext |
| `--scenario_ids` | `0` | 病例 id：`0,1,2` / `0-9` / `all` |
| `--runs` | `30` | 每个病例重复次数 N |
| `--doctor_llm` | `deepseek-v4-pro` | 医生模型 |
| `--patient_llm` | `deepseek-v4-flash` | 病人模型 |
| `--measurement_llm` | `deepseek-v4-pro` | 检查结果模型 |
| `--moderator_llm` | `deepseek-v4-pro` | 裁判模型（仅 semantic / grade 时用） |
| `--bucketing` | `exact` | `exact` 原始字符串 / `semantic` 语义合并 |
| `--grade_correctness` | 关 | 是否判对错 |
| `--temperatures` | `0.05` | 温度，逗号分隔可多值 |
| `--total_inferences` | `20` | 每次问诊最多几轮 |
| `--per_call_sleep` | `0.5` | 每步间隔（秒），缓解限流 |
| `--concurrency` | `1` | 并发数，1=串行 |
| `--out_dir` | 自动 | 输出目录 |
| `--verbose` | 关 | 打印每轮对话细节 |

---

## 7. 两个需要知道的机制

### 强制收尾（`forced` 字段）
医生最多问 `--total_inferences`（默认 20）轮。如果问满 20 轮还没主动给出诊断，脚本会**额外发一次请求逼它给一个最终诊断**，并标记 `forced=true`。这保证每次都有诊断结果，但 `forced=true` 的诊断是在不同 prompt 下产生的，分析时可以单独留意。
> 注：这个强制兜底是本项目新增的，原版 AgentClinic 没有。

### 为什么慢 / 怎么提速
`deepseek-v4-pro` 偏慢，而且医生经常一路问满 20 轮，单次问诊可能要几分钟。提速办法：
- 降 `--total_inferences`（如 10）→ 轮数减半，耗时大致减半；
- 辅助模型用 flash（patient 默认已是 flash）；
- `--per_call_sleep 0` 去掉间隔；
- 开 `--concurrency`。

---

## 8. 常见问题

- **结果会上传/覆盖吗？** 每个病例跑完立即写盘（增量保存），中途崩溃已跑的不丢；同名 `out_dir` 会覆盖同名 case 文件，建议每次换 `--out_dir`。
- **要花多少钱/时间？** 大致 = 病例数 × runs × 温度数 ×（每次问诊约 10–40 次 API 调用）。先用 `--runs 3` 小样估算再放大。
- **看不到实时输出？** 加 `-u`（`python -u ...`），否则重定向到文件时会缓冲。

---

如有问题问 Junhan。

*（文档更新于 2026-05-31）*
