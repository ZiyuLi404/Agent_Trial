# anchor_compare — 锚定回归（C）

## 这个模块干嘛
判断版本升级后**到底是"模型行为变了"还是"病例变难了"**。

做法：挑一批**固定的标准病人（anchor case）**，让 baseline 模型和 candidate 模型在**同一批 case** 上各重复跑几遍，**不看对错、不需要金标准**，只比两个模型的输出像不像。

> 锁住病例这个变量 → 表现还有差别，就只能是模型本身变了。

## 4 个行为漂移指标（全不依赖金标准）
| 指标 | 看什么 |
|------|--------|
| Top-1 一致率 | 最终诊断变没变（可用 moderator 判临床等价） |
| 候选 Jaccard | 鉴别诊断候选清单变没变 |
| 证据 Jaccard | 关注的关键证据变没变 |
| JS 散度 | 多次重跑的诊断分布漂移 |

为支持比较，DoctorAgent 在 `output_format="anchor_compare"` 下强制结构化输出：`DIAGNOSIS READY` + `CANDIDATES:` + `KEY EVIDENCE:`。

## 怎么跑
```bash
# 从仓库根目录
python anchor_compare/anchor_compare.py --eval_mode anchor_compare \
    --baseline_doctor_llm deepseek-v4-flash --candidate_doctor_llm deepseek-v4-pro \
    --num_scenarios 10 --runs_per_case 3 ...
```

## 与其他模块的关系
- 概念上是 **② `trial` 在线准确率**通道之外的**第二条正交评估通道**（行为漂移），两者联合解读（见根目录 `notes/design.md`）。
- 和 **G/I（embedding / ICD 散度）** 目标相近——都在"不靠金标准量诊断漂移"，后者是更成熟的离线版本。

## 状态 / 重大技术债
- ⚠️ **早期搭建、当前搁置**：2026-05-27 创建后基本未动，研究重心已转向 G/I/H。保留待定，非死代码。
- ⚠️ **它是引擎的整段复制分支**（~1338 行，自带一份 query_model / 各 Agent / ScenarioLoader，**不** `import AgentClinic`）。且其 `query_model` 停在旧版（`temperature=0`、`max_tokens=200`，易截断结构化输出）。
  - 理想改法：改成 `from AgentClinic.agentclinic import ...`，仅用薄子类保留 `output_format="anchor_compare"` 的结构化差异。
  - **用户已决定本轮先不动**。详见根目录 `REFACTOR_PLAN.md` §6。
- ✅ 已从根目录收进本模块；`DATA_DIR` 修正为指向仓库根的 `AgentClinic/`（移目录后仍能找到数据集）。
