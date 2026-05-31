# Agent Trial 设计 Note

源自 2026-05-22 白板讨论（见 `notes/May22_1.jpg`、`notes/May22_2.jpg`）。
本文档梳理 Agent Trial 的整体评估框架，并标记哪些已实现、哪些待办。

---

## 1. 核心问题

我们要回答的不是"AgentClinic 跑得准不准"，而是 **"当 doctor agent 版本升级时，怎么稳定、可比、可追溯地评估这次升级？"**

升级会带来两类信号混在一起：

- **真实病例 mix 的变化**：新一批病例本身就更难/更易
- **版本本身的行为变化**：模型/prompt/工具升级带来的差异

单看线上准确率分不开这两件事。所以设计两条**正交**的评估通道。

---

## 2. 两条评估通道

```
线上流（run_trial.py）：     V1 [20 patients] → V2 [15] → V3 [25] → ...
                                  │                │             │
                                  └────────────────┴─────────────┘
                                                 │
                                  在每次版本切换点触发
                                                 │
                                                 ▼
锚定回归（agentclinic_anchor_compare.py）：
                              固定 a 个 anchor case 在 V1 / V2 / V3 上各重跑
                              得到 V1↔V2、V2↔V3 的纯行为漂移指标
```

| 通道    | 文件                                  | 评估对象              | 评估指标                                        | 是否依赖金标准 |
| ----- | ----------------------------------- | ----------------- | ------------------------------------------- | ------- |
| 线上准确率 | `run_trial.py` + `trial_manager.py` | 实时进来的真实病例流        | Acc（每个 version_id 一个）                       | ✓       |
| 锚定回归  | `agentclinic_anchor_compare.py`     | 固定 anchor case 集合 | Top-1 一致率 / 候选 Jaccard / 证据 Jaccard / JS 散度 | ✗       |

两条数据**联合解读**：

- 线上 Acc 掉了 + 锚定指标稳定 → 病例 mix 变了，不是模型回归
- 线上 Acc 稳定 + 锚定指标漂移 → 模型行为变了，只是恰好运气没掉准确率
- 两者都掉 → 模型回归，需要回滚

---

## 3. Anchor case 抽样策略（白板讨论结论）

用混淆矩阵的视角选 anchor 数量 `a`：

|            | 漂移真实存在     | 无漂移 |
| ---------- | ---------- | --- |
| anchor 检出  | TP         | FP  |
| anchor 未检出 | **FN（漏检）** | TN  |

- `a` 太小 → FN 高，真实漂移被漏掉
- `a` 太大 → 成本爆炸，跑不起

**目标：找最小的 `a`，让 FN 控制在可接受范围。**

当前默认（待真实数据验证）：

- `--num_scenarios`：anchor case 数（白板示意 a ≈ 10–20）
- `--runs_per_case`：每版每病例重复次数（默认 3，用来估计行为分布）

---

## 4. 跨版本上下文：三种 borrow 方案

V1 → V2 切换的瞬间，V2 该不该看到 V1 留下的对话历史？白板上列了三条路线：

| 方案         | 含义                                             | 现状                    |
| ---------- | ---------------------------------------------- | --------------------- |
| **borrow** | V2 继承 V1 的 `agent_hist`（所有问诊 context + 阶段诊断结果） | ❌ 未实现                 |
| **开关眼**    | 把 V1 时期跑过的所有 patient 在 V2 上**全部重跑**一遍，拉齐基线     | ❌ 未实现                 |
| **只看当前版本** | 每个纪元独立，新版本从头开始                                 | ✓ `run_trial.py` 当前行为 |

**对比维度**：三种方案下 V2 的 Acc 差异 → 揭示 "borrow 是否真的能加速新版本上线"。

> 这是 Agent Trial 相对原版 AgentClinic 的**核心新能力**——原引擎每个病例完全独立，没有跨病例/跨版本的 context。

---

## 5. 当前实现状态

### 已实现

- ✓ 线上流 + 版本纪元 + JSONL 日志（`run_trial.py` / `version_manager.py` / `logger.py`）
- ✓ 1:1 块随机分组（`randomization.py`）
- ✓ 锚定回归主流程（`agentclinic_anchor_compare.py::run_anchor_comparison`）
- ✓ 4 个行为指标（Top-1 / 候选 Jaccard / 证据 Jaccard / JS 散度）
- ✓ DoctorAgent 在 `output_format="anchor_compare"` 下强制结构化输出

### 待实现

- ❌ borrow 方案（cross-version history injection）
- ❌ "开关眼" 方案（旧版本病例在新版本上批量重跑）
- ❌ Anchor 自动触发：当 `run_trial.py` 检测到 `--new_version` 时自动跑一次 `run_anchor_comparison`
- ❌ Anchor 选择算法：从 `trial_log.jsonl` 历史里自动挑能区分版本的 case 作为下一轮 anchor

### 已知不一致 / 待对齐

- `trial_manager.run_case` 没接 ReviewerAgent，但 README 描述了"两医生工作流"
- `agentclinic_anchor_compare.py` 是引擎代码的复制分支，不 import `AgentClinic/agentclinic.py`，未来引擎改动需要同步两份
- `randomization._block` 不跨进程持久化
- `version_detect.py` 用 Bing 搜索 hash 做版本变更监听，未接入主流程

---

## 6. 下一步实验计划

> 白板上的数字（V1 = 20 patients、V2 = 15、V3 = 25 …）是为了讲清楚思路**人为编的示意值**，真实实验尚未开始。

需要先确定的参数：

- 每个纪元真实接多少病例（影响线上 Acc 的统计功效）
- Anchor 集合规模 `a` 和重跑次数 `runs_per_case`
- borrow 方案的具体接口（注入到 PatientAgent 还是 DoctorAgent？只注入最近一次 history 还是滚动累积？）

第一轮可跑的最小实验：

1. 用现有 `run_trial.py` 跑 V1 = `deepseek-v4-flash` 20 个病例 → 切到 V2 = `deepseek-v4-pro` 再跑 20 个
2. 切版本的同时用 `agentclinic_anchor_compare.py` 跑一次锚定回归（baseline=V1，candidate=V2，anchor=10 case，runs=3）
3. 看：
   - 线上 Acc 差异是否显著
   - 锚定四指标是否触发 "behaviorally_equivalent = False"
   - 两者方向是否一致

结果回填到本文档第 6 节末尾。
