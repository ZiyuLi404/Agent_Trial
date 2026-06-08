"""
diagnosis_distribution.py
=========================

实验：固定一个 case，让 doctor / patient（同一模型，如 DeepSeek）反复模拟问诊，
收集 doctor agent 的最终诊断，逼近其在该 case 上的「诊断结果概率分布」。

设计要点（与 anchor_compare 实验无关，互不影响）：

1. 重复粒度 = 整段问诊独立重跑。
   每个样本都从头跑一遍完整的 doctor-patient-measurement 多轮对话，
   捕捉所有随机来源（患者回答 + 医生推理），最贴近「反复测试同一 case」。

2. 诊断结果 = doctor 最后一句 "DIAGNOSIS READY:" 后的自由文本
   （跑满轮数仍未给出则用 force_doctor_final_diagnosis 逼一个）。
   每个样本保留完整原始对话 full_dialogue，便于以后任意方式再分析。

3. 默认只积累原始数据，全程不调用 moderator：
   - bucketing="exact"（默认）：按归一化字符串（大小写/空白无关）分桶计数，
     得到原始结果分布。表述不同但语义相同的诊断保留为不同桶，不做合并。
   - grade_correctness=False（默认）：不判对错。金标准诊断仅作参考写入文件
     （字段 correct_diagnosis_reference），供以后用 moderator 或其他方法再分析。
   需要时可开 --bucketing semantic（moderator 语义合并）或 --grade_correctness（moderator 判对错）。

4. 温度：默认 0.05（引擎生产值，最初观察到现象的温度）。--temperatures 传多个值即做
   温度对照（如 0,0.05,0.7）：同一批 case 在多个温度下各跑一遍，用于拆开
   『服务端底噪 (temp=0)』与『采样发散 (temp>0)』两类随机来源。
   注意：分布是相对温度定义的，不存在温度无关的『真实分布』；temp=0 残余的
   变化来自服务端非确定性（MoE 路由 / batch 拼接 / 浮点累加），非平稳，仅作底噪基准。
   引擎 query_model 已支持 temperature；本脚本在每个温度下覆盖 engine.DEFAULT_TEMPERATURE，
   不影响 anchor_compare / run_trial 的默认 0.05 行为。

用法示例（默认：积累原始数据，不调 moderator）：
    python diagnosis_distribution.py --scenario_ids 0,1,2 --runs 30 \
        --doctor_llm deepseek-v4-pro --patient_llm deepseek-v4-pro \
        --out_dir results/dist_run1

结果目录结构：
    out_dir/temp_0.05/case_0.json ...    # 每个 (温度, case) 一个 JSON（含完整对话 + 分布）
    out_dir/summary.json                 # 每行 = (温度, case) 的统计量
末尾控制台打印汇总表（distinct / entropy / mode）。增量写盘，中途崩溃已跑结果不丢。
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 自动加载 .env（与 run_trial.py / anchor_compare 一致）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed; .env will not be auto-loaded. "
          "Run `pip install python-dotenv` or set keys via export/CLI flags.")

# 直接复用原引擎，避免再复制一份代码（design.md 里提到的维护痛点）。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import AgentClinic.agentclinic as engine  # noqa: E402
from AgentClinic.agentclinic import (  # noqa: E402
    ScenarioLoaderMedQA,
    ScenarioLoaderMedQAExtended,
    ScenarioLoaderNEJM,
    ScenarioLoaderNEJMExtended,
    PatientAgent,
    DoctorAgent,
    MeasurementAgent,
    extract_diagnosis_text,
    force_doctor_final_diagnosis,
    compare_results,
    normalize_answer,
    load_doctor_prompt_template,
    load_icd10cm_dictionary,
    parse_icd10cm_from_diagnosis,
    validate_icd10cm_output,
    get_diagnosis_bucket_key,
    resolve_local_path,
)

_LOADERS = {
    "MedQA": ScenarioLoaderMedQA,
    "MedQA_Ext": ScenarioLoaderMedQAExtended,
    "NEJM": ScenarioLoaderNEJM,
    "NEJM_Ext": ScenarioLoaderNEJMExtended,
}


# ---------------------------------------------------------------------------
# 单次完整问诊模拟（= 一个样本）
# ---------------------------------------------------------------------------
def run_single_simulation(
    scenario,
    doctor_llm,
    patient_llm,
    measurement_llm,
    total_inferences=20,
    per_call_sleep=0.5,
    verbose=False,
    label="",
    print_lock=None,
    doctor_prompt_template=None,
    diagnosis_output_format="text",
    icd10cm_dict=None,
    validate_icd10cm=False,
):
    """跑一遍完整的 doctor-patient-measurement 对话，返回 doctor 的最终诊断。

    复刻 agentclinic.main() 的单 scenario 内层循环，但只保留 doctor/patient/measurement，
    不接 reviewer（本实验聚焦 doctor 自己的诊断分布）。

    label/print_lock 用于并发时给每行对话加 run 前缀并保证整行不撕裂。
    """
    def vprint(msg):
        if not verbose:
            return
        line = f"{label}{msg}"
        if print_lock is not None:
            with print_lock:
                print(line, flush=True)
        else:
            print(line, flush=True)

    meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
    patient_agent = PatientAgent(scenario=scenario, bias_present="None", backend_str=patient_llm)
    doctor_agent = DoctorAgent(
        scenario=scenario,
        bias_present="None",
        backend_str=doctor_llm,
        max_infs=total_inferences,
        doctor_prompt_template=doctor_prompt_template,
    )

    pi_dialogue = ""
    full_dialogue = ""
    doctor_final_response = None
    forced = False
    num_infs = 0

    for _inf_id in range(total_inferences):
        if _inf_id == total_inferences - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis using DIAGNOSIS READY: [diagnosis].\n"

        doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue)
        num_infs = _inf_id + 1
        vprint(f"Doctor: {doctor_dialogue}")
        full_dialogue += "Doctor: " + doctor_dialogue + "\n"

        if "DIAGNOSIS READY" in doctor_dialogue.upper():
            doctor_final_response = doctor_dialogue
            break

        if "REQUEST TEST" in doctor_dialogue:
            pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
            vprint(f"Measurement: {pi_dialogue}")
            full_dialogue += "Measurement: " + pi_dialogue + "\n"
            patient_agent.add_hist(pi_dialogue)
        else:
            pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
            vprint(f"Patient: {pi_dialogue}")
            full_dialogue += "Patient: " + pi_dialogue + "\n"
            meas_agent.add_hist(pi_dialogue)

        if per_call_sleep > 0:
            time.sleep(per_call_sleep)

    # 医生从未给出 DIAGNOSIS READY → 逼一个最终结论
    if doctor_final_response is None:
        doctor_final_response = force_doctor_final_diagnosis(
            doctor_llm=doctor_llm,
            scenario=scenario,
            full_dialogue=full_dialogue,
        )
        forced = True
        vprint(f"Doctor (forced): {doctor_final_response}")

    diagnosis_text = extract_diagnosis_text(doctor_final_response)
    result = {
        "diagnosis_text": diagnosis_text,
        "final_diagnosis": diagnosis_text,
        "diagnosis_output_format": diagnosis_output_format,
        "raw_response": normalize_answer(doctor_final_response),
        "forced": forced,
        "num_infs": num_infs,
        "full_dialogue": full_dialogue,
    }

    if diagnosis_output_format == "icd10cm":
        parsed_icd = parse_icd10cm_from_diagnosis(doctor_final_response)
        if validate_icd10cm and icd10cm_dict is not None:
            parsed_icd = validate_icd10cm_output(parsed_icd, icd10cm_dict)
        if not parsed_icd.get("icd10cm_valid", False):
            vprint(f"Warning: invalid or unrecognized ICD-10-CM output: {doctor_final_response}")
        result.update(parsed_icd)
        result["final_diagnosis"] = (
            parsed_icd.get("icd10cm_dictionary_code")
            or parsed_icd.get("icd10cm_code")
            or diagnosis_text
        )

    return result


# ---------------------------------------------------------------------------
# 分桶 / 聚类
# ---------------------------------------------------------------------------
def _same_disease(a, b, moderator_llm):
    """用 moderator 判断两个诊断文本是否同一种病。复用引擎的 compare_results。"""
    if normalize_answer(a).lower() == normalize_answer(b).lower():
        return True
    ans = compare_results(a, b, moderator_llm, None)
    return ans.strip().startswith("yes")


def bucket_semantic(diagnoses, gold, moderator_llm, verbose=False):
    """语义贪心聚类。先按精确字符串预聚合以省 LLM 调用，再跨不同字符串语义合并。

    返回 (clusters, sample_cluster_idx)：
      clusters: [{"canonical", "members": {variant: count}, "count"}]
      sample_cluster_idx: 每个样本所属簇下标，与输入 diagnoses 等长
    """
    # 1) 精确去重，得到 distinct 字符串及其计数与首次出现顺序
    distinct = OrderedDict()  # norm_text -> {"display", "count", "members_idx": [...]}
    for idx, d in enumerate(diagnoses):
        key = normalize_answer(d).lower()
        if key not in distinct:
            distinct[key] = {"display": normalize_answer(d), "count": 0, "idx": []}
        distinct[key]["count"] += 1
        distinct[key]["idx"].append(idx)

    # 2) 跨 distinct 字符串语义合并
    clusters = []  # 每个 cluster: {"rep": str, "members": OrderedDict, "idx": []}
    for key, info in distinct.items():
        placed = False
        for cl in clusters:
            if _same_disease(info["display"], cl["rep"], moderator_llm):
                cl["members"][info["display"]] = info["count"]
                cl["idx"].extend(info["idx"])
                placed = True
                break
        if not placed:
            clusters.append({
                "rep": info["display"],
                "members": OrderedDict({info["display"]: info["count"]}),
                "idx": list(info["idx"]),
            })

    # 3) 选每个簇里出现次数最多的变体作为 canonical 名称
    sample_cluster_idx = [None] * len(diagnoses)
    out = []
    for ci, cl in enumerate(clusters):
        canonical = max(cl["members"].items(), key=lambda kv: kv[1])[0]
        for i in cl["idx"]:
            sample_cluster_idx[i] = ci
        out.append({
            "canonical": canonical,
            "members": dict(cl["members"]),
            "count": sum(cl["members"].values()),
        })
    return out, sample_cluster_idx


def bucket_exact(diagnoses):
    """仅按归一化字符串精确分桶，不调用 LLM。"""
    distinct = OrderedDict()
    sample_cluster_idx = [None] * len(diagnoses)
    order = []
    for idx, d in enumerate(diagnoses):
        disp = normalize_answer(d)
        key = disp.lower()
        if key not in distinct:
            distinct[key] = {"canonical": disp, "count": 0, "ci": len(order)}
            order.append(key)
        distinct[key]["count"] += 1
        sample_cluster_idx[idx] = distinct[key]["ci"]
    out = [{"canonical": distinct[k]["canonical"],
            "members": {distinct[k]["canonical"]: distinct[k]["count"]},
            "count": distinct[k]["count"]} for k in order]
    return out, sample_cluster_idx


def shannon_entropy_bits(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# ---------------------------------------------------------------------------
# 单 case：跑 N 次并统计分布
# ---------------------------------------------------------------------------
def run_case_distribution(
    scenario,
    scenario_id,
    runs,
    doctor_llm,
    patient_llm,
    measurement_llm,
    moderator_llm,
    bucketing="exact",
    grade_correctness=False,
    total_inferences=20,
    per_call_sleep=0.5,
    verbose=False,
    concurrency=1,
    doctor_prompt_template=None,
    diagnosis_output_format="text",
    icd10cm_dict=None,
    validate_icd10cm=False,
):
    """固定 case 跑 N 次，收集 doctor 最终诊断，按原始字符串得到结果分布。

    默认 bucketing="exact" + grade_correctness=False → 全程不调用 moderator，
    只积累原始记录（含完整对话）。金标准诊断仅作参考写入文件，不参与判定。
    需要时可开 grade_correctness 或 bucketing="semantic" 启用 moderator。
    """
    gold = scenario.diagnosis_information()  # 仅作参考记录，不参与分桶/判定
    samples = [None] * runs
    print_lock = threading.Lock()
    done_counter = {"n": 0}

    def one_run(r):
        t0 = time.time()
        sim = run_single_simulation(
            scenario=scenario,
            doctor_llm=doctor_llm,
            patient_llm=patient_llm,
            measurement_llm=measurement_llm,
            total_inferences=total_inferences,
            per_call_sleep=per_call_sleep,
            verbose=verbose,
            label=f"[c{scenario_id} r{r}] ",
            print_lock=print_lock,
            doctor_prompt_template=doctor_prompt_template,
            diagnosis_output_format=diagnosis_output_format,
            icd10cm_dict=icd10cm_dict,
            validate_icd10cm=validate_icd10cm,
        )
        sample = {
            "run": r,
            "diagnosis_text": sim["diagnosis_text"],
            "final_diagnosis": sim["final_diagnosis"],
            "diagnosis_output_format": sim["diagnosis_output_format"],
            "raw_response": sim["raw_response"],
            "forced": sim["forced"],
            "num_infs": sim["num_infs"],
            "full_dialogue": sim["full_dialogue"],   # 完整原始对话记录
            "elapsed_sec": round(time.time() - t0, 1),
        }
        # Carry ICD fields through if present
        for key in ("icd10cm_code", "icd10cm_code_normalized", "icd10cm_label",
                    "raw_final_diagnosis", "icd10cm_valid",
                    "icd10cm_dictionary_code", "icd10cm_dictionary_label"):
            if key in sim:
                sample[key] = sim[key]

        # 可选：与金标准比对（默认关，关时不调 moderator）
        if grade_correctness:
            sample["correct"] = compare_results(
                sim["diagnosis_text"], gold, moderator_llm, None
            ).strip().startswith("yes")

        with print_lock:
            done_counter["n"] += 1
            mark = ""
            if grade_correctness:
                mark = "✓ " if sample["correct"] else "✗ "
            display_diag = sim["final_diagnosis"]
            print(f"  run {r + 1:>2}/{runs}  done [{done_counter['n']:>2}/{runs}]  "
                  f"{sample['elapsed_sec']:>5.1f}s  infs={sim['num_infs']:>2} "
                  f"forced={'Y' if sim['forced'] else 'N'}  | {mark}{display_diag}",
                  flush=True)
        return r, sample

    if concurrency <= 1:
        for r in range(runs):
            _, samples[r] = one_run(r)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(one_run, r) for r in range(runs)]
            for f in as_completed(futs):
                r, sample = f.result()
                samples[r] = sample

    bucket_keys = [get_diagnosis_bucket_key(s) for s in samples]

    # ---- 分桶（默认 exact：按原始字符串，不做语义合并、不调 moderator）----
    if bucketing == "exact":
        clusters, sample_ci = bucket_exact(bucket_keys)
    elif bucketing == "semantic":
        clusters, sample_ci = bucket_semantic(bucket_keys, gold, moderator_llm, verbose=verbose)
    else:
        raise ValueError(f"Unknown bucketing: {bucketing}")

    n = len(samples)
    for cl in clusters:
        cl["prob"] = cl["count"] / n if n else 0.0
    # 按出现次数降序
    order = sorted(range(len(clusters)), key=lambda ci: -clusters[ci]["count"])
    remap = {old: new for new, old in enumerate(order)}
    clusters = [clusters[old] for old in order]
    sample_ci = [remap[ci] for ci in sample_ci]
    for i, s in enumerate(samples):
        s["bucket"] = clusters[sample_ci[i]]["canonical"]

    counts = [cl["count"] for cl in clusters]
    ent = shannon_entropy_bits(counts)
    max_ent = math.log2(len(clusters)) if len(clusters) > 1 else 0.0

    result = {
        "scenario_id": scenario_id,
        "correct_diagnosis_reference": gold,   # 仅参考，未用于任何判定
        "runs": runs,
        "bucketing": bucketing,
        "graded": grade_correctness,
        "doctor_llm": doctor_llm,
        "patient_llm": patient_llm,
        "measurement_llm": measurement_llm,
        "moderator_llm": (moderator_llm if (grade_correctness or bucketing == "semantic") else None),
        "temperature": engine.DEFAULT_TEMPERATURE,
        "total_inferences": total_inferences,
        "diagnosis_output_format": diagnosis_output_format,
        "validate_icd10cm": validate_icd10cm,
        "num_distinct_buckets": len(clusters),
        "entropy_bits": ent,
        "normalized_entropy": (ent / max_ent) if max_ent > 0 else 0.0,
        "mode_diagnosis": clusters[0]["canonical"] if clusters else None,
        "mode_prob": clusters[0]["prob"] if clusters else 0.0,
        "distribution": clusters,
        "samples": samples,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if grade_correctness:
        result["p_correct"] = sum(1 for s in samples if s.get("correct")) / n if n else 0.0
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_scenario_ids(spec, num_available):
    """支持 '0,1,2' 或 '0-9' 或 'all'。"""
    spec = spec.strip()
    if spec == "all":
        return list(range(num_available))
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return ids


def parse_str_list(spec: str) -> list:
    """Parse a comma-separated list of strings, e.g. 'deepseek-v4-pro, deepseek-v4-flash'."""
    return [s.strip() for s in spec.split(",") if s.strip()]


def parse_prompt_styles(spec: str, prompt_json_path: str) -> list:
    """Parse comma-separated style names or 'all' (expands to every key in the JSON file)."""
    spec = spec.strip()
    if spec == "all":
        path = resolve_local_path(prompt_json_path)
        with path.open("r", encoding="utf-8") as f:
            text = "\n".join(
                line for line in f
                if not line.lstrip().startswith("//")
            )
        prompt_bank = json.loads(text)
        return list(prompt_bank.keys())
    return [s.strip() for s in spec.split(",") if s.strip()]


def main():
    # 行缓冲：即使 stdout 重定向到文件，也能逐行落盘，便于实时 tail -f 监控。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="固定 case 反复模拟，估计 doctor agent 诊断结果的概率分布。")
    parser.add_argument("--dataset", type=str, default="MedQA", choices=list(_LOADERS.keys()))
    parser.add_argument("--scenario_ids", type=str, default="0",
                        help="case id，如 '0,1,2' / '0-9' / 'all'")
    parser.add_argument("--runs", type=int, default=10, help="每个 case 重复模拟次数 N")
    parser.add_argument("--doctor_llm", type=str, default="deepseek-v4-pro",
                        help="Doctor LLM(s), comma-separated, e.g. 'deepseek-v4-pro' / 'deepseek-v4-pro, deepseek-v4-flash'")
    parser.add_argument("--patient_llm", type=str, default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", type=str, default="deepseek-v4-pro")
    parser.add_argument("--moderator_llm", type=str, default="deepseek-v4-pro",
                        help="裁判模型，仅在 --bucketing semantic 或 --grade_correctness 时才会被调用")
    parser.add_argument("--bucketing", type=str, default="exact",
                        choices=["exact", "semantic"],
                        help="exact（默认）：按原始字符串分桶，不调 moderator、不做语义合并；"
                             "semantic：用 moderator 合并语义等价诊断（会调用 moderator）。")
    parser.add_argument("--grade_correctness", action="store_true",
                        help="是否用 moderator 判断每个诊断对错（默认关）。关时金标准仅作参考写入文件。")
    parser.add_argument("--temperatures", type=str, default="0.05",
                        help="温度扫描列表，逗号分隔。默认 0.05（引擎生产值，你最初观察到现象的温度）。"
                             "传多个值即做温度对照，如 '0,0.05,0.7'：同一批 case 在每个温度下各跑一遍，"
                             "用于拆开『服务端底噪 (temp=0)』与『采样发散 (temp>0)』。"
                             "作用于 doctor/patient/measurement/moderator 所有 LLM 调用。")
    parser.add_argument("--total_inferences", type=int, default=20)
    parser.add_argument("--per_call_sleep", type=float, default=0.5,
                        help="对话内每步之间的 sleep，缓解限流")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="单个 case 内并发跑多少次重复模拟。默认 1=串行（不并行）。"
                             "调大可显著提速（N 次重跑彼此独立），verbose 下各 run 对话会以 [cX rY] 前缀交错。")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join("results", "diag_dist_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--verbose", action="store_true", help="打印每轮对话细节")
    parser.add_argument("--deepseek_api_key", type=str, default=None)
    parser.add_argument("--openai_api_key", type=str, default=None)
    # Doctor prompt
    parser.add_argument("--doctor_prompt_json", type=str, default="doctor_prompts.json",
                        help="Path to JSON file containing doctor prompt templates")
    parser.add_argument("--doctor_prompt_style", type=str, default="default",
                        help="Prompt style(s), comma-separated or 'all', e.g. 'default' / 'default, icd10cm' / 'all'")
    # ICD-10-CM validation
    parser.add_argument("--icd10cm_jsonl", type=str, default="icd10cm_2026.jsonl",
                        help="Path to ICD-10-CM JSONL code dictionary. Relative paths resolved relative to agentclinic.py.")
    parser.add_argument("--validate_icd10cm", action="store_true",
                        help="Validate ICD-10-CM diagnosis outputs against the JSONL dictionary.")
    # Convenience aliases for case range / runs / output dir
    args = parser.parse_args()

    if args.deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    runs = args.runs
    out_dir = args.out_dir

    temperatures = [float(t) for t in args.temperatures.split(",") if t.strip() != ""]
    doctor_llms = parse_str_list(args.doctor_llm)
    prompt_styles = parse_prompt_styles(args.doctor_prompt_style, args.doctor_prompt_json)
    multi_combo = len(doctor_llms) > 1 or len(prompt_styles) > 1

    loader = _LOADERS[args.dataset]()
    scenario_ids = parse_scenario_ids(args.scenario_ids, loader.num_scenarios)
    scenario_ids = [i for i in scenario_ids if 0 <= i < loader.num_scenarios]

    # Load ICD-10-CM dictionary once (shared across all styles)
    icd10cm_dict = None
    if args.validate_icd10cm:
        icd10cm_dict = load_icd10cm_dictionary(args.icd10cm_jsonl)
        print(f"Loaded {len(icd10cm_dict)} ICD-10-CM codes from {args.icd10cm_jsonl}")

    os.makedirs(out_dir, exist_ok=True)
    bar = "═" * 70
    print(bar)
    print(f" diagnosis distribution  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" out_dir       : {out_dir}")
    print(f" dataset       : {args.dataset}   cases: {scenario_ids}")
    print(f" runs/case     : {runs}      temperatures: {temperatures}")
    print(f" bucketing     : {args.bucketing}   grade_correctness: {args.grade_correctness}")
    print(f" doctor_llms   : {doctor_llms}")
    print(f" prompt_styles : {prompt_styles}")
    print(f" patient       : {args.patient_llm}   measurement: {args.measurement_llm}")
    if args.validate_icd10cm:
        print(f" icd10cm       : {args.icd10cm_jsonl} ({len(icd10cm_dict)} codes)")
    if args.bucketing == "semantic" or args.grade_correctness:
        print(f" moderator     : {args.moderator_llm}  (启用)")
    else:
        print(f" moderator     : 未启用（不判对错、不语义合并）")
    print(bar)

    summary = []
    for temp in temperatures:
        engine.DEFAULT_TEMPERATURE = temp
        temp_dir = os.path.join(out_dir, f"temp_{temp}")
        os.makedirs(temp_dir, exist_ok=True)
        if len(temperatures) > 1:
            print(f"\n############  temperature = {temp}  ############")

        for doctor_llm in doctor_llms:
            llm_slug = doctor_llm.replace("/", "_").replace(" ", "_")

            for prompt_style in prompt_styles:
                # Load the template for this specific style
                doctor_prompt_template = None
                try:
                    doctor_prompt_template = load_doctor_prompt_template(
                        args.doctor_prompt_json, prompt_style
                    )
                except FileNotFoundError as e:
                    if prompt_style == "default" and args.doctor_prompt_json == "doctor_prompts.json":
                        print(f"Warning: {e}. Using built-in default.")
                    else:
                        raise

                diagnosis_output_format = "text"
                if doctor_prompt_template is not None:
                    diagnosis_output_format = doctor_prompt_template.get(
                        "diagnosis_output_format", "text"
                    )

                # When running multiple LLM/style combos, place results in subdirs
                case_dir = (
                    os.path.join(temp_dir, llm_slug, prompt_style)
                    if multi_combo
                    else temp_dir
                )
                os.makedirs(case_dir, exist_ok=True)

                if multi_combo:
                    print(f"\n{'─'*70}")
                    print(f"  llm={doctor_llm}  style={prompt_style}  format={diagnosis_output_format}")
                    print(f"{'─'*70}")

                for sid in scenario_ids:
                    scenario = loader.get_scenario(id=sid)
                    gold = scenario.diagnosis_information()
                    print(f"\n┌─ Case {sid}  (temp={temp}, llm={doctor_llm}, style={prompt_style}, N={runs})")
                    print(f"│  gold(参考): {gold}")
                    result = run_case_distribution(
                        scenario=scenario,
                        scenario_id=sid,
                        runs=runs,
                        doctor_llm=doctor_llm,
                        patient_llm=args.patient_llm,
                        measurement_llm=args.measurement_llm,
                        moderator_llm=args.moderator_llm,
                        bucketing=args.bucketing,
                        grade_correctness=args.grade_correctness,
                        total_inferences=args.total_inferences,
                        per_call_sleep=args.per_call_sleep,
                        verbose=args.verbose,
                        concurrency=args.concurrency,
                        doctor_prompt_template=doctor_prompt_template,
                        diagnosis_output_format=diagnosis_output_format,
                        icd10cm_dict=icd10cm_dict,
                        validate_icd10cm=args.validate_icd10cm,
                    )

                    case_path = os.path.join(case_dir, f"case_{sid}.json")
                    with open(case_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                    print(f"│")
                    print(f"│  诊断分布 (N={runs}, distinct={result['num_distinct_buckets']}, "
                          f"entropy={result['entropy_bits']:.3f} bits):")
                    for cl in result["distribution"]:
                        grade = ""
                        if args.grade_correctness:
                            grade = "✓ " if cl.get("is_correct") else "✗ "
                        print(f"│    {cl['prob'] * 100:5.1f}%  ({cl['count']:>2}/{runs})  "
                              f"{grade}{cl['canonical']}")
                    if args.grade_correctness:
                        print(f"│  P(correct) = {result['p_correct']:.3f}")
                    print(f"└─ saved → {case_path}")

                    row = {
                        "temperature": temp,
                        "doctor_llm": doctor_llm,
                        "prompt_style": prompt_style,
                        "diagnosis_output_format": diagnosis_output_format,
                        "scenario_id": sid,
                        "correct_diagnosis_reference": result["correct_diagnosis_reference"],
                        "num_distinct_buckets": result["num_distinct_buckets"],
                        "entropy_bits": result["entropy_bits"],
                        "normalized_entropy": result["normalized_entropy"],
                        "mode_diagnosis": result["mode_diagnosis"],
                        "mode_prob": result["mode_prob"],
                    }
                    if args.grade_correctness:
                        row["p_correct"] = result["p_correct"]
                    summary.append(row)
                    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
                        json.dump({"config": vars(args), "rows": summary}, f, ensure_ascii=False, indent=2)

    # ---- 汇总表 ----
    print(f"\n{bar}")
    print(" 汇总 (distinct / entropy" + (" / Pcorrect" if args.grade_correctness else "") + ")")
    print(bar)
    pcol = "  Pcorr" if args.grade_correctness else ""
    print(f"{'temp':>6}  {'llm':<22}  {'style':<18}  {'case':>4}  {'distinct':>8}  {'entropy':>8}{pcol}   mode")
    for row in summary:
        pcell = f"  {row.get('p_correct', 0):>5.2f}" if args.grade_correctness else ""
        print(f"{row['temperature']:>6}  {row['doctor_llm']:<22}  {row['prompt_style']:<18}  "
              f"{row['scenario_id']:>4}  {row['num_distinct_buckets']:>8}  "
              f"{row['entropy_bits']:>8.3f}{pcell}   "
              f"{row['mode_diagnosis']} ({row['mode_prob'] * 100:.0f}%)")

    print(f"\nDone. 每个 (温度, llm, style, case) 一个 JSON + summary.json 已写入 {out_dir}")


if __name__ == "__main__":
    main()
