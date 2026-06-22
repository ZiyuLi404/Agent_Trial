# generate_diagnosis_distribution — 诊断分布生成（F · 上游生产者）

## 这个模块干嘛
**固定一个病例，让 doctor/patient 反复模拟问诊很多遍**，收集每次的最终诊断，逼近该 case 上的「诊断结果概率分布」——看 AI 输出到底稳不稳、分布多散。

它是**上游生产者**：跑出的 `case_X.json` 是下游 **G `embedding_similarity`** 和 **I `kl_js_divergence`** 的输入数据。

## 它其实是个通用 runner（两个维度）
`diagnosis_distribution.py` 可在 `模型 × prompt风格` 上扫，两种用法：
- **固定 prompt、看输出稳不稳** → "诊断分布"实验
- **换 `--doctor_prompt_style`、看不同人设的诊断漂移** → **prompt 实验**（prompt 库在 `AgentClinic/doctor_prompts.json`，7 种人设）

## 文件
| 文件 | 作用 |
|------|------|
| `diagnosis_distribution.py` | 主 runner。直接复用引擎（`import AgentClinic.agentclinic`，不复制代码） |
| `make_dist_report.py` | 把结果渲成英文文字报告（从归档收回，本就是 F 的配套） |
| `run_gpt_data.sh` | 批量编排：逐 case 跑某些模型、组装成 `<tag>/case_X.json` 文件夹 |

## 输出格式
每个 `case_X.json` 含 `scenario_id` / `runs` / `distribution`（按桶计数）/ `samples`（每次的 `diagnosis_text` + `full_dialogue`）/ `entropy_bits` 等。

## 怎么跑
```bash
# 从仓库根目录
python generate_diagnosis_distribution/diagnosis_distribution.py \
    --doctor_llm deepseek-v4-flash --scenario_ids 0,1,2 --runs 30 ...
# 多模型一次扫（GPT 只是另一个模型值，不需要单独脚本）：
python generate_diagnosis_distribution/diagnosis_distribution.py --doctor_llm gpt-5.5,gpt-5-mini ...
```

## 重构记录 / 待办
- ✅ 从根目录收进本模块；bootstrap 改为把**仓库根**加进 `sys.path`（子目录下仍能 `import AgentClinic`）。
- ✅ `make_dist_report.py` 从归档收回；其 `RESULTS_DIR` 指向仓库根的 `results/`。
- ⏳ **数据落点**：让产物落到 `results/generate_diagnosis_distribution/<model>/case_X.json`（取代手工打包的 `Data*.zip`）。这是 G 改读目录、Data*.zip 退役的配套大改，见 `REFACTOR_PLAN.md` §3.2。
- ⚠️ `run_gpt_data.sh` 末尾的 `zip -r Data_gpt.zip` 步骤在"取消 zip"后**作废**，届时删该步即可（编排主体保留）。
