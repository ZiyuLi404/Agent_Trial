# AgentClinic — 基础引擎（①·地基）

## 这个模块干嘛
一个**模拟诊所**：AI 医生（DoctorAgent）、AI 病人（PatientAgent）、AI 化验师（MeasurementAgent）互相对话看病，最后医生给出诊断。**项目里所有要"真的跑问诊"的模块都建在它之上。**

源自上游 [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic)，本仓库在其基础上做了少量增量（见下「我们的改动」）。

## 关键文件
| 文件 | 作用 |
|------|------|
| `agentclinic.py` | 核心引擎：各 Agent、`ScenarioLoader*`、`query_model`、`compare_results` 等 |
| `doctor_prompts.json` | 医生 prompt 库（7 种人设：default / icd10cm / 5 种风格）← prompt 实验的料 |
| `agentclinic_medqa.jsonl` 等 ×4 | MedQA / NEJM（及扩展）病例数据集 |
| `icd10cm_2026.jsonl` + `icd10cm_codes/` | **ICD-10-CM 字典的唯一来源**（引擎校验 + I `kl_js_divergence` 共用） |

## 作为包被复用
其他模块统一以**绝对包导入**引用它：
```python
from AgentClinic.agentclinic import DoctorAgent, query_model, ScenarioLoaderMedQA, ...
```
被 `trial` / `deployment_replay` / `generate_diagnosis_distribution` 依赖。
（`anchor_compare` 目前是它的**复制分支**，尚未改成 import——见该模块 README。）

## 我们相对上游的改动
- `doctor_prompt_template` / `load_doctor_prompt_template` 机制（2026-06-08 加）：支持从 `doctor_prompts.json` 切换医生人设，向后兼容（默认行为同原版）。
- ICD-10-CM 校验相关函数。
- GPT-5.x reasoning 模型适配、DeepSeek reasoning_content 处理等。

## 维护原则
这是**地基**，要稳、少动。改动尽量**additive、向后兼容**，避免破坏上面一层层的实验。
ICD 字典只保留这一份（其他模块引用此处），避免多份漂移。
