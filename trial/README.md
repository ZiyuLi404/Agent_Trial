# trial — 在线试验框架（②）

## 这个模块干嘛
论文主线：把 **doctor agent 当成会升级的版本**（V1→V2→V3…），一个病例一个病例地喂进引擎，随机分到 **对照组 / 实验组**，把每例结果写进日志，用来衡量**版本升级后的准确率变化**。

建在 **① `AgentClinic` 引擎**之上——引擎负责"跑一次问诊"，本模块负责"按试验设计组织成千上万次问诊并记录"。

## 文件
| 文件 | 作用 |
|------|------|
| `run_trial.py` | CLI 主入口，编排版本管理 / 流式跑 case / 日志 / 准确率展示 |
| `trial_manager.py` | `stream_cases` 顺序流式取 case，`run_case` 包一次问诊 |
| `version_manager.py` | 读写版本纪元状态（`current_version.json`） |
| `randomization.py` | 1:1 块随机，给每个 case 分 `control` / `treatment` |
| `logger.py` | 把每例结果追加进 `trial_log.jsonl`；`log_deployment_case` 供 E 用 |
| `current_version.json` | 当前版本指针（属模块状态，路径已锁定到本目录，不随 CWD） |

## 怎么跑
两种方式都行（模块已做成包 + 入口带 sys.path 引导）：
```bash
# 从仓库根目录
python -m trial.run_trial --new_version --version_id v1 --model_name deepseek-v4-flash ...
# 或直接
python trial/run_trial.py --new_version --version_id v1 ...
```

## 与其他模块的关系
- 依赖 **① `AgentClinic`**（`from AgentClinic.agentclinic import ...`）。
- 被 **E `deployment_replay`** 复用（`from trial.trial_manager import ...`、`from trial.logger import ...`）。

## 重构记录 / 待办
- ✅ 五件套从根目录收进 `trial/` 包；内部改绝对包导入 `from trial.X import`。
- ✅ `current_version.json` 路径锁定到本目录（不随运行目录漂移）。
- ⏳ `trial_log.jsonl` 等产物按蓝图迁到 `results/trial/`（目前仍写到运行目录）。
- ⚠️ 已知债：`run_case` 未接 ReviewerAgent（README 主线描述的"两医生工作流"与实现不符）；`randomization._block` 不跨进程持久化。详见根目录 `REFACTOR_PLAN.md` §6。
