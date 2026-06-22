# deployment_replay — 部署重放 / 混合估计（E）

## 这个模块解决的死结
升级到新版医生 V2 后，**V2 只能看到从现在起的新病人**；可你手里有 V1 时代攒的一堆老病例。要公平评价 V2，最好让它把老病例也看一遍，**但老病人早走了，叫不回来重新问诊**。

> E 研究：**老病人回不来，怎么估 V2 在他们身上的表现？**

## 三种"补考"老病例的办法
| 办法 | 怎么做 | 现实可行？ |
|------|--------|-----------|
| **Oracle**（理想答案） | 让 V2 从头重新问诊每个老病人 | ❌ 只有模拟里能做（counterfactual 真值） |
| **Replay**（重放病历） | 给 V2 看老病历的对话记录，只凭记录下诊断 | ✅ 现实能做（档案都在） |
| **Hybrid**（混合） | 老病例用 replay + 新病例用实时，拼成整体估计 | ✅ 现实能做 |

**核心问题**：用 replay（现实唯一能做的）代替 oracle（理想但做不到的），到底损失多少准确率？
- replay ≈ oracle → 现实中就能用"重放老病历"快速给新版补全评价，不用干等新病人
- replay 差很多 → 只看记录丢太多信息（V2 本会问不同问题、开不同化验），捷径走不通

## 文件
| 文件 | 作用 |
|------|------|
| `deployment_timeline.py` | 入口：按"诊所一天天来病人"的时间线，做 paired shadow 评估，串起下面三件 |
| `replay_evaluator.py` | 重放：V2 只读历史对话记录下诊断 |
| `oracle_evaluator.py` | 理想：V2 从头重新问诊（复用 `trial.trial_manager.run_case`） |
| `hybrid_estimator.py` | 把 replay（老）+ 实时（新）按 case 数加权成混合估计 |

## 怎么跑
```bash
# 从仓库根目录
python -m deployment_replay.deployment_timeline ...
# 或直接
python deployment_replay/deployment_timeline.py ...
```

## 与其他模块的关系
- 依赖 **② `trial`**（`from trial.trial_manager import ...`、`from trial.logger import ...`）和 **① `AgentClinic`**。
- 兄弟：和 **H `history_borrowing`** 同属"新版本数据太少怎么借历史"——E 纵向借（自己版本的老病例，靠重放），H 横向借（相似模型）。属 design 的 borrow 主线。

## 状态 / 待办
- ⚠️ **早期搭建、当前搁置**：2026-05-28 加入、06-08 刷新过，之后未推进；本工作区**从未跑出过产物**（无 `deployment_log.jsonl` / `transcripts/`）。保留待续，非死代码。
- ✅ 从根目录收进 `deployment_replay/` 包；import 改 `trial.*` / `deployment_replay.*` 绝对包形式。
- ⏳ 产物按蓝图落到 `results/deployment_replay/`。详见根目录 `REFACTOR_PLAN.md`。
