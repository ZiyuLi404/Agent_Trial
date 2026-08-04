import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from AgentClinic.agentclinic import extract_diagnosis_text
from agent_system_adapters.agentclinic import AgentClinicAdapter
from change_generators.harnesses import HarnessArtifact
from change_generators.skills import SkillArtifact
from change_generators.skills.optimizers.skillopt_lite.samples import export_samples
from change_generators.skills.optimizers.skillopt_lite.splits import (
    build_split_manifest,
    load_split_config,
    select_case_ids,
)
from trial.trial_manager import LOADERS


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPLIT_CONFIG = REPO_ROOT / "experiments" / "agentclinic_skillopt" / "splits.toml"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def dialogue_to_messages(dialogue: str) -> list[dict]:
    markers = list(re.finditer(r"(?m)^(Doctor|Patient|Measurement):[ \t]*", dialogue or ""))
    messages = []
    role_map = {"Doctor": "assistant", "Patient": "user", "Measurement": "tool"}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(dialogue)
        content = dialogue[marker.end():end].strip()
        if content:
            messages.append({
                "role": role_map[marker.group(1)],
                "speaker": marker.group(1),
                "content": content,
            })
    return messages


def build_result_row(
    *,
    dataset: str,
    split: str,
    case_id: int,
    scenario,
    diagnosis,
    correctness: bool,
    dialogue: str,
    meta: dict,
    contract_dry_run: bool,
) -> tuple[dict, list[dict]]:
    task_id = f"{dataset}-{case_id:05d}"
    question = str(scenario.examiner_information())
    correct_text = str(scenario.diagnosis_information())
    response = "" if diagnosis is None else str(diagnosis)
    predicted = extract_diagnosis_text(response)
    messages = dialogue_to_messages(dialogue)
    n_turns = sum(message["speaker"] == "Doctor" for message in messages)
    tests_requested = sum(
        message["speaker"] == "Doctor" and "REQUEST TEST" in message["content"]
        for message in messages
    )
    doctor_retry_count = int(meta.get("doctor_retry_count", 0) or 0)
    doctor_empty_response = bool(meta.get("doctor_empty_response"))
    interaction_calls = sum(
        message["speaker"] in {"Patient", "Measurement"} for message in messages
    )
    moderator_calls = int("DIAGNOSIS READY" in response)
    observed_model_calls = 0 if contract_dry_run else (
        n_turns
        + doctor_retry_count
        + int(doctor_empty_response)
        + interaction_calls
        + moderator_calls
    )

    if contract_dry_run:
        fail_reason = "contract-dry-run: no model API call was made"
    elif meta.get("doctor_empty_response"):
        fail_reason = meta.get("backend_error_message") or "doctor returned an empty response"
    elif not correctness:
        fail_reason = f"incorrect diagnosis: predicted {predicted!r}; expected {correct_text!r}"
    else:
        fail_reason = ""

    evaluation_message = {
        "role": "system",
        "speaker": "Evaluator",
        "content": (
            f"[EVALUATION RESULT]\nCorrect: {bool(correctness)}\n"
            f"Predicted diagnosis: {predicted}\nGold diagnosis: {correct_text}"
        ),
    }
    conversation = messages + [evaluation_message]
    row = {
        "id": task_id,
        "scenario_id": case_id,
        "dataset": dataset,
        "split": split,
        "task_type": dataset.lower(),
        "task_description": question,
        "question": question,
        "correct_text": correct_text,
        "predicted_answer": predicted,
        "response": response,
        "hard": int(bool(correctness)),
        # AgentClinic's current moderator is binary. Keep soft equal to hard
        # until a separately validated continuous clinical grader is added.
        "soft": float(bool(correctness)),
        "agent_ok": not bool(meta.get("doctor_empty_response")) and not contract_dry_run,
        "fail_reason": fail_reason,
        "n_turns": n_turns,
        "tests_requested": tests_requested,
        "observed_model_calls": observed_model_calls,
        "backend": {
            "raw_doctor_response_empty": bool(meta.get("raw_doctor_response_empty")),
            "doctor_retry_count": doctor_retry_count,
            "doctor_empty_response": doctor_empty_response,
            "backend_error_message": str(meta.get("backend_error_message", "") or ""),
            "reasoning_content_present": bool(meta.get("reasoning_content_present")),
        },
        "trajectory": conversation,
        "variant": meta.get("variant", {}),
        "phase": "contract_dry_run" if contract_dry_run else "evaluation",
    }
    return row, conversation


def write_prediction_artifacts(
    output_dir: Path,
    task_id: str,
    conversation: list[dict],
    system_prompt: str,
    task_prompt: str,
) -> None:
    prediction_dir = output_dir / "predictions" / task_id
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.joinpath("conversation.json").write_text(
        json.dumps(conversation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prediction_dir.joinpath("target_system_prompt.txt").write_text(
        system_prompt, encoding="utf-8"
    )
    prediction_dir.joinpath("target_user_prompt.txt").write_text(
        task_prompt, encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SkillOpt-Lite-compatible AgentClinic evaluator"
    )
    parser.add_argument("--skill", required=True, help="Candidate skill markdown path")
    parser.add_argument("--harness", default=None, help="Optional fixed harness TOML path")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--split_config", default=str(DEFAULT_SPLIT_CONFIG))
    parser.add_argument("--dataset", default="MedQA",
                        choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"])
    parser.add_argument("--eval_limit", type=int, default=0,
                        help="Cases sent to the model; 0 means the full split")
    parser.add_argument("--limit", type=int, default=20,
                        help="Maximum sample markdown files; 0 means all")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed when eval_limit is smaller than the split")
    parser.add_argument("--doctor_llm", default="deepseek-v4-pro")
    parser.add_argument("--patient_llm", default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", default="deepseek-v4-flash")
    parser.add_argument("--moderator_llm", default="deepseek-v4-flash")
    parser.add_argument("--total_inferences", type=int, default=20)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract_dry_run", action="store_true",
                        help="Write the full output contract without calling model APIs")
    return parser


def evaluate(args: argparse.Namespace) -> dict:
    skill = SkillArtifact.load(args.skill)
    harness = HarnessArtifact.load(args.harness) if args.harness else None
    adapter = AgentClinicAdapter(skill, harness)

    loader_cls = LOADERS[args.dataset]
    loader = loader_cls()
    split_config = load_split_config(args.split_config)
    split_manifest = build_split_manifest(
        args.dataset, loader.num_scenarios, split_config
    )
    selected_ids = select_case_ids(
        split_manifest, args.split, args.eval_limit, args.seed
    )

    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else (
        REPO_ROOT
        / "results"
        / "skillopt_workspaces"
        / "agentclinic"
        / safe_name(args.doctor_llm)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        workspace / ".skillopt" / "_eval_run" / f"{args.split}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.joinpath("skill.md").write_text(skill.content + "\n", encoding="utf-8")

    results_path = output_dir / "results.jsonl"
    existing_rows = []
    done_ids = set()
    if results_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"results.jsonl already exists: {results_path}; use --resume or a new output_dir"
            )
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing_rows.append(row)
                done_ids.add(str(row["id"]))

    run_manifest = {
        **split_manifest,
        "selected_split": args.split,
        "selected_ids": selected_ids,
        "sample_seed": args.seed,
        "skill": skill.to_dict(include_content=False),
        "harness": harness.to_dict() if harness else None,
        "models": {
            "doctor": args.doctor_llm,
            "patient": args.patient_llm,
            "measurement": args.measurement_llm,
            "moderator": args.moderator_llm,
        },
        "contract_dry_run": bool(args.contract_dry_run),
    }
    output_dir.joinpath("split_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config = {
        "doctor_llm": args.doctor_llm,
        "patient_llm": args.patient_llm,
        "measurement_llm": args.measurement_llm,
        "moderator_llm": args.moderator_llm,
        "total_inferences": args.total_inferences,
    }
    new_rows = []
    with results_path.open("a", encoding="utf-8") as results_file:
        for completed, case_id in enumerate(selected_ids, start=1):
            task_id = f"{args.dataset}-{case_id:05d}"
            if task_id in done_ids:
                continue
            scenario = loader.get_scenario(id=case_id)
            doctor = adapter.build_doctor(scenario, config)
            system_prompt = doctor.system_prompt()

            if args.contract_dry_run:
                diagnosis = None
                correctness = False
                dialogue = ""
                meta = {"variant": adapter.variant_metadata()}
            else:
                diagnosis, correctness, dialogue, meta = adapter.evaluate_case(
                    scenario, config
                )

            row, conversation = build_result_row(
                dataset=args.dataset,
                split=args.split,
                case_id=case_id,
                scenario=scenario,
                diagnosis=diagnosis,
                correctness=correctness,
                dialogue=dialogue,
                meta=meta,
                contract_dry_run=args.contract_dry_run,
            )
            write_prediction_artifacts(
                output_dir,
                task_id,
                conversation,
                system_prompt,
                str(scenario.examiner_information()),
            )
            results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            results_file.flush()
            new_rows.append(row)
            print(
                f"[agentclinic-skillopt] {completed}/{len(selected_ids)} "
                f"id={task_id} hard={row['hard']} soft={row['soft']:.3f}",
                flush=True,
            )

    all_rows = existing_rows + new_rows
    sample_counts = export_samples(all_rows, workspace, args.limit)
    hard = sum(float(row.get("hard", 0)) for row in all_rows) / max(len(all_rows), 1)
    soft = sum(float(row.get("soft", 0)) for row in all_rows) / max(len(all_rows), 1)
    summary = {
        "n": len(all_rows),
        "hard": hard,
        "soft": soft,
        "observed_model_calls": sum(
            int(row.get("observed_model_calls", 0) or 0) for row in all_rows
        ),
        "backend_health": {
            "rows_with_retries": sum(
                int((row.get("backend") or {}).get("doctor_retry_count", 0) > 0)
                for row in all_rows
            ),
            "total_doctor_retries": sum(
                int((row.get("backend") or {}).get("doctor_retry_count", 0) or 0)
                for row in all_rows
            ),
            "empty_response_failures": sum(
                int(bool((row.get("backend") or {}).get("doctor_empty_response")))
                for row in all_rows
            ),
        },
        "samples": sample_counts,
        "results_path": str(results_path),
        "workspace": str(workspace),
        "output_dir": str(output_dir),
    }
    output_dir.joinpath("metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results: hard={hard:.4f} soft={soft:.4f}")
    print(
        f"Samples: failed={sample_counts['failed']} passed={sample_counts['passed']} "
        f"workspace={workspace / '.skillopt' / 'samples'}"
    )
    return summary


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
