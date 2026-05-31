# CARC 迁移清单与启动指引

把 `/Users/junhan/26summer/Agent_Trial` 迁到 USC CARC（Discovery 集群）通过 OnDemand
（<https://ondemand.carc.usc.edu>）继续开发和跑实验。

> **项目性质提醒**：本项目是 **纯 API 调用型实验**（DeepSeek / OpenAI / Anthropic /
> Replicate），不在本地做模型推理或训练，**不需要 GPU、不需要 CUDA、不需要大显存**。
> 上 HPC 的真正动机是：
>
> 1. 长跑（一次 trial 几百个 case，每个 case 上百次 API 调用，要跑几小时甚至几天）不再
>    占用本地机器；
> 2. login node 会限制长进程 → 用 SLURM 把它丢到计算节点，断网/合盖电脑也不影响；
> 3. 多版本（V1 / V2 / V3 …）和锚定回归可以并行多个 job 一起跑。
>
> 所以 **CPU partition 就够用**，资源规划相对简单。

---

## 1. 当前 workspace 盘点

### 1.1 要上传的文件

| 类别 | 路径 | 是否上传 | 备注 |
|---|---|---|---|
| 主入口 | `run_trial.py` | ✅ | 线上流主程序 |
| 主入口 | `agentclinic_anchor_compare.py` | ✅ | 锚定回归主程序 |
| 模块 | `trial_manager.py` / `version_manager.py` / `randomization.py` / `logger.py` / `version_detect.py` | ✅ | 全部上传 |
| 引擎 | `AgentClinic/agentclinic.py` | ✅ | 核心引擎 |
| 数据 | `AgentClinic/agentclinic_medqa.jsonl` 等 4 个 jsonl | ✅ | ~850KB，体积小直接传 |
| 配置 | `requirements.txt` | ✅ | 依赖清单 |
| 配置 | `.env.example` | ✅ | 模板，可入仓 |
| 配置 | `current_version.json` | ⚠️ | 可上传，但建议在 CARC 上重新初始化（避免本地状态污染） |
| 文档 | `README.md` / `notes/*` | ✅ | 含本文件 |
| 密钥 | `.env` | ❌ **不要走 git** | 在 CARC 上手动新建，见 §4 |
| 输出 | `anchor_compare_debug.json` / `trial_log.jsonl` | ❌ | 实验输出，在 CARC 上重新生成 |
| 缓存 | `AgentClinic/__pycache__/` / `.DS_Store` | ❌ | 跳过 |

### 1.2 总体积

代码 + 数据 ≈ **< 2 MB**，OnDemand 网页上传或 `scp` 一次就完事。

### 1.3 依赖（`requirements.txt`）

```
openai>=1.0.0
anthropic
transformers
replicate
accelerate
torch
python-dotenv
```

注意：`transformers` / `torch` / `accelerate` 在当前 trial 流程里 **没有实际使用**
（项目只走云端 API，没有 HuggingFace 本地模型）。在 CARC 上为了省装包时间和磁盘配额，
可以临时新建一个精简版 `requirements-carc.txt`：

```
openai>=1.0.0
anthropic
replicate
python-dotenv
numpy
scipy
```

（`numpy` / `scipy` 是 anchor_compare 算 JS 散度等指标可能用到的隐式依赖；如果运行时
报缺包就再加。）

---

## 2. 上传到 CARC

### 2.1 推荐：OnDemand Files（最简单）

1. 浏览器登录 <https://ondemand.carc.usc.edu>（USC NetID `junhanwu`）。
2. 顶部菜单 **Files → Home Directory**。
3. 在 `/home1/junhanwu/` 下新建目录 `Agent_Trial`。
4. 进入该目录，点 **Upload** → 选 `/Users/junhan/26summer/Agent_Trial` 整个文件夹
   （或者打个 zip 上传更快）：
   ```bash
   # 本地先打包，排除掉密钥和输出
   cd /Users/junhan/26summer
   zip -r Agent_Trial.zip Agent_Trial \
       -x "*.env" "*__pycache__*" "*.DS_Store" \
          "Agent_Trial/anchor_compare_*.json" \
          "Agent_Trial/trial_log.jsonl" \
          "Agent_Trial/.git/*"
   ```
   再把 `Agent_Trial.zip` 上传到 OnDemand，**Extract** 即可。

### 2.2 命令行替代：`scp` / `rsync`

```bash
# 走 jump host 或者直接 ssh discovery
rsync -av --progress \
  --exclude '.env' --exclude '__pycache__' --exclude '.DS_Store' \
  --exclude 'anchor_compare_*.json' --exclude 'trial_log.jsonl' \
  --exclude '.git' \
  /Users/junhan/26summer/Agent_Trial/ \
  junhanwu@discovery.usc.edu:/home1/junhanwu/Agent_Trial/
```

### 2.3 （可选）做免密 SSH

把本地公钥追加到 CARC：

```bash
ssh junhanwu@discovery.usc.edu "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
cat ~/.ssh/id_rsa.pub | ssh junhanwu@discovery.usc.edu \
  "cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

---

## 3. 在 CARC 上选 module 和建环境

### 3.1 查 module（先开 OnDemand 的 **Clusters → Discovery Shell Access**）

```bash
module purge
module avail python      # 看支持的 python 版本
module avail gcc         # 看可用 gcc
```

CARC Discovery 上常见组合（按 USC CARC 文档 <https://www.carc.usc.edu/user-guides>）：

```bash
module load gcc/11.3.0
module load python/3.9.12
```

> 项目本地用 `python=3.10`（见 README）。3.9 与 3.10 在本项目这种纯 API 调用代码里
> **完全兼容**（没用 3.10 独有的 `match` / 新类型语法）。如果 CARC 也提供
> `python/3.10.x`，优先 load 3.10；没有就用 3.9.12。

### 3.2 建环境：推荐 **conda env**（与 README 一致，且 CARC 官方推荐）

CARC 一般预装了 Miniconda 或 Mamba，先确认：

```bash
module load conda  # 部分集群叫 anaconda3 / miniforge
which conda
```

建环境：

```bash
# 把 conda envs 放到 project / scratch，不要塞满 home（home 通常只有 100GB）
mkdir -p /project/<your_PI_account>/junhanwu/conda-envs   # 如果有 PI 配额
# 或者放 scratch（注意 scratch 会定期清理）：
mkdir -p /scratch1/junhanwu/conda-envs

conda create -p /scratch1/junhanwu/conda-envs/agenttrial python=3.10 -y
conda activate /scratch1/junhanwu/conda-envs/agenttrial

cd /home1/junhanwu/Agent_Trial
pip install -r requirements.txt
# 或省事版：
# pip install openai anthropic replicate python-dotenv numpy scipy
```

如果你确认完全不会用 transformers / torch，**强烈建议**装精简版——`torch` 单独 2GB+，
首次安装非常慢，对本项目纯属浪费。

### 3.3 venv 备选

如果 conda 不顺：

```bash
module load python/3.10.x  # 或 3.9.12
python -m venv /scratch1/junhanwu/venvs/agenttrial
source /scratch1/junhanwu/venvs/agenttrial/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. 配置 API 密钥（在 CARC 上，**不要把本地 .env 上传**）

登录到 CARC 之后：

```bash
cd /home1/junhanwu/Agent_Trial
cp .env.example .env
nano .env   # 或 vim
# 填入：
#   DEEPSEEK_API_KEY=...
#   OPENAI_API_KEY=...
#   ANTHROPIC_API_KEY=...
#   REPLICATE_API_TOKEN=...
chmod 600 .env
```

`run_trial.py` 和 `agentclinic_anchor_compare.py` 启动时都会 `load_dotenv()`，所以
不需要在 SLURM 脚本里额外 `export`。

> 注意 Discovery 计算节点**可以访问外网**（绝大多数 USC CARC partition 都允许出站
> HTTPS），所以 OpenAI / DeepSeek API 调用没有问题。如果跑 job 报 connection refused，
> 联系 CARC support 申请白名单。

---

## 5. 资源需求估计

| 维度 | 估算 | 理由 |
|---|---|---|
| CPU | 1 core 足够 | 串行 IO 等待为主 |
| 内存 | 2–4 GB | jsonl scenario 小，对话历史轻 |
| GPU | **不需要** | 全部走云端 API |
| 时长（单 epoch 20 case，每 case 20 turn） | ~1–3 小时 | 取决于 API 延迟 + `time.sleep(1.0)` |
| 时长（anchor_compare 10 case × 3 runs × 2 versions） | ~1–2 小时 | 同上 |
| 磁盘 | < 1 GB | 输出 jsonl 一次 trial 几 MB 量级 |

→ 用 **`main` / `epyc-64` 这种 CPU partition** 即可（具体名字以 `sinfo` 实际显示为准）。

---

## 6. SLURM 提交脚本

新建 `/home1/junhanwu/Agent_Trial/job_trial.slurm`：

```bash
#!/bin/bash
#SBATCH --job-name=agenttrial_v1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wjh7000@gmail.com

set -euo pipefail

# 切到项目目录
cd /home1/junhanwu/Agent_Trial
mkdir -p logs

# 环境
module purge
module load gcc/11.3.0
module load conda             # 或 module load python/3.10.x
conda activate /scratch1/junhanwu/conda-envs/agenttrial

# 跑一个 V1 calibration epoch
python run_trial.py \
  --new_version --version_id v1 --model_name deepseek-v4-flash \
  --prompt_version p1 --tool_version t1 \
  --control_llm deepseek-v4-flash \
  --doctor_llm  deepseek-v4-flash \
  --patient_llm deepseek-v4-flash \
  --measurement_llm deepseek-v4-flash \
  --moderator_llm   deepseek-v4-flash \
  --dataset MedQA --num_cases 20 --total_inferences 10
```

锚定回归同理（`job_anchor.slurm`）：

```bash
#!/bin/bash
#SBATCH --job-name=anchor_v1_vs_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

set -euo pipefail
cd /home1/junhanwu/Agent_Trial
mkdir -p logs

module purge
module load gcc/11.3.0
module load conda
conda activate /scratch1/junhanwu/conda-envs/agenttrial

python agentclinic_anchor_compare.py \
  --baseline_llm deepseek-v4-flash \
  --candidate_llm deepseek-v4-pro \
  --num_scenarios 10 \
  --runs_per_case 3 \
  --dataset MedQA
# 实际参数以 agentclinic_anchor_compare.py argparse 为准，必要时 --help
```

提交：

```bash
sbatch job_trial.slurm
sbatch job_anchor.slurm
squeue -u junhanwu          # 看队列
scancel <jobid>             # 取消
tail -f logs/agenttrial_v1_*.out
```

---

## 7. 交互式开发（在 OnDemand 网页里）

**写代码**和**调试**不要在 login node 跑训练（CARC 会杀进程）。两个推荐入口：

### 7.1 OnDemand → **Interactive Apps → Jupyter**

- partition: `main`（CPU）
- cpus: 1, memory: 4G, time: 2–4 小时
- 启动后开 terminal cell 一样能跑 `python run_trial.py ...`，但**短跑 / debug 用就好**，
  长跑还是 sbatch。

### 7.2 OnDemand → **Interactive Apps → code-server (VS Code)**

- 这就是 Discovery 屏蔽掉 Remote-SSH 之后的官方替代品，浏览器里跑一个完整的 VS Code，
  直接挂在计算节点上，可以一边改代码一边在内建 terminal 里 `sbatch`。
- 申请同样资源（1 CPU / 4G / 4h）。
- 工作目录指向 `/home1/junhanwu/Agent_Trial`。

### 7.3 纯 SSH 调试（不开 OnDemand）

```bash
ssh junhanwu@discovery.usc.edu
salloc --partition=main --cpus-per-task=1 --mem=4G --time=01:00:00
# 进入计算节点后再 activate env 跑
```

---

## 8. 落地 checklist（按顺序勾掉）

1. [ ] 本地 `zip` 项目（排除 `.env` / `__pycache__` / 输出 jsonl / `.git`）
2. [ ] OnDemand Files 上传 → 解压到 `/home1/junhanwu/Agent_Trial/`
3. [ ] OnDemand Shell：`module load gcc/11.3.0 python/3.10.x`（或 conda）
4. [ ] 在 `/scratch1/junhanwu/conda-envs/agenttrial` 建 conda env
5. [ ] `pip install -r requirements.txt`（或精简版）
6. [ ] `cp .env.example .env` 并填 4 个 API key，`chmod 600 .env`
7. [ ] 写 `job_trial.slurm` / `job_anchor.slurm`，建 `logs/` 目录
8. [ ] `sbatch job_trial.slurm` 跑 V1 calibration 验证 pipeline
9. [ ] `tail -f logs/agenttrial_v1_*.out` 看输出
10. [ ] 跑通后再切 V2 / 启动 anchor_compare

---

## 9. 常见坑

- **home 满了**：CARC home 通常 100GB，conda env 别装 home，扔 `/scratch1/` 或
  `/project/<PI>/`。
- **scratch 被清**：`/scratch1/` 一般 14 天不访问会清，重要输出务必拷回 home 或本地。
- **API 调用挂住**：`trial_manager.run_case` 里有 `time.sleep(1.0)`，外加 LLM API 偶
  尔超时。SLURM `--time` 留足余量（建议 `--time=04:00:00` 起步）。
- **DeepSeek / OpenAI 限速**：在 CARC 上跑也会撞 rate limit，必要时在 `agentclinic.py`
  的调用层加重试或拉长 sleep。
- **断网/中断**：`trial_log.jsonl` 是 append 模式，job 中断不会丢历史；重启时只要
  不再 `--new_version`，会继续在同一个 epoch 里追加。
- **路径**：本项目用相对路径（`AgentClinic/agentclinic_medqa.jsonl` 等），SLURM
  脚本里务必先 `cd /home1/junhanwu/Agent_Trial`。

---

## 10. 参考链接

- CARC 用户指南：<https://www.carc.usc.edu/user-guides>
- OnDemand 门户：<https://ondemand.carc.usc.edu>
- AgentClinic 上游：<https://github.com/SamuelSchmidgall/AgentClinic>
