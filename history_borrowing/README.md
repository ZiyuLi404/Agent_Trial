# history_borrowing — 历史借用 / 性能估计（H）

## 这个模块干嘛
当某个模型在某个病例桶上**测的样本太少、准确率不可信**时，从**"行为长得像的其他模型"那里借数据**来修正它的估计——让少量样本也能估得更准。

属于**离线分析**：不跑问诊、不碰引擎，只读已经跑出来的准确率和相似度数据。

## 输入
- `accuracy_by_25_cases.csv` —— 每个桶的准确率 + 总金标准数（由 `accuracy_summary.py` 生成）
- `*_similarity_matrix.csv` —— 模型间的相似度矩阵（来自 **G `embedding_similarity`** 的产出）

## 怎么算（核心公式）
把相似度转成距离 `d(A,B)=1-sim(A,B)`，再让每个模型按距离加权借用同伴的准确率：

```
theta_borrowed_j = alpha * theta_j + (1-alpha) * Σ_{i≠j} w_ij * theta_i
```

`alpha` 控制"信自己 vs 信同伴"，`w_ij` 由距离经 `lambda` 加权。

## 脚本（5 个 = 5 个真步骤，按顺序）
| 脚本 | 步骤 |
|------|------|
| `accuracy_summary.py` | 从 groundtruth 目录汇总每桶准确率 → `accuracy_by_25_cases.csv` |
| `history_borrowing.py` | 单次借用估计（给定 alpha/lambda） |
| `run_all_orders.py` | 对 24 种"桶↔模型"排列各跑一遍 `history_borrowing.py` |
| `train_borrow_params.py` | 全局调一对 (alpha, lambda)，最小化所有排列/模型的平均 MAE |
| `visualize_borrow_params.py` | 把结果画成 dashboard |

## 怎么跑
```bash
# 从仓库根目录运行
python history_borrowing/accuracy_summary.py --groundtruth_dir history_borrowing/groundtruth
python history_borrowing/history_borrowing.py --accuracy_csv history_borrowing/accuracy_by_25_cases.csv ...
python history_borrowing/run_all_orders.py
python history_borrowing/train_borrow_params.py
python history_borrowing/visualize_borrow_params.py --source diagnosis
```

## 与其他模块的关系
- 上游：**G `embedding_similarity`** 产的相似度矩阵是本模块的输入。
- 兄弟：和 **E `deployment_replay`** 同属"新版本数据太少怎么借历史"——H 横向借（从相似模型），E 纵向借（从自己版本的老病例）。

## 重构待办
- ⏳ 产物 / 中间数据按全局蓝图迁到 `results/history_borrowing/`（目前仍留在本目录，argparse 默认路径也指向本目录）。详见根目录 `REFACTOR_PLAN.md` §5。
