"""Run AgentClinic episodes while ReflACT treats only the skill as trainable."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_runtime(agentclinic_repo: Path):
    import sys

    repo_text = str(agentclinic_repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    try:
        from dotenv import load_dotenv

        load_dotenv(agentclinic_repo / ".env")
    except ImportError:
        pass

    from AgentClinic.agentclinic import ScenarioLoaderMedQA
    from skill_harness.artifacts import SkillArtifact
    from skill_harness.common.agentclinic import AgentClinicAdapter
    from skill_harness.common.agentclinic.evaluation import (
        build_result_row,
        write_prediction_artifacts,
    )

    return {
        "loader_cls": ScenarioLoaderMedQA,
        "adapter_cls": AgentClinicAdapter,
        "artifact_cls": SkillArtifact,
        "build_result_row": build_result_row,
        "write_prediction_artifacts": write_prediction_artifacts,
    }


def _failure_row(item: dict, exc: Exception) -> tuple[dict, list[dict]]:
    message = f"AgentClinic rollout exception: {type(exc).__name__}: {exc}"
    conversation = [{"role": "system", "content": message}]
    return (
        {
            "id": str(item["id"]),
            "scenario_id": int(item["scenario_id"]),
            "split": str(item["split"]),
            "task_type": "medqa",
            "task_description": "AgentClinic medical diagnosis",
            "question": "AgentClinic medical diagnosis",
            "correct_text": "",
            "predicted_answer": "",
            "response": "",
            "hard": 0,
            "soft": 0.0,
            "agent_ok": False,
            "fail_reason": message,
            "n_turns": 0,
            "observed_model_calls": 0,
        },
        conversation,
    )


def _run_one(payload: dict) -> tuple[int, dict, list[dict], str, str]:
    """Run one case in a process so AgentClinic globals and cwd stay isolated."""
    position = int(payload["position"])
    item = dict(payload["item"])
    repo = Path(payload["agentclinic_repo"])
    output_dir = Path(payload["output_dir"])
    runtime = _load_runtime(repo)
    skill_content = str(payload["skill_content"])
    skill_hash = hashlib.sha256(skill_content.encode("utf-8")).hexdigest()
    skill = runtime["artifact_cls"](
        skill_id="diagnostic_reasoning",
        version=f"reflact-{skill_hash[:12]}",
        content=skill_content,
        sha256=skill_hash,
        path=output_dir / "skill.md",
        generated_by="original-skillopt-reflact",
    )
    adapter = runtime["adapter_cls"](skill_artifact=skill, harness_artifact=None)
    config = {
        "doctor_llm": payload["doctor_llm"],
        "patient_llm": payload["patient_llm"],
        "measurement_llm": payload["measurement_llm"],
        "moderator_llm": payload["moderator_llm"],
        "total_inferences": int(payload["total_inferences"]),
    }

    with _working_directory(repo):
        loader = runtime["loader_cls"]()
        scenario = loader.get_scenario(int(item["scenario_id"]))
    system_prompt = adapter.build_doctor(scenario, config).system_prompt()
    task_prompt = str(scenario.examiner_information())
    try:
        if payload["contract_dry_run"]:
            diagnosis, correctness, dialogue, meta = "", False, "", {
                "variant": adapter.variant_metadata(),
                "measurement": {"mode": "generative", "request_count": 0},
                "interaction_backend": {
                    "patient_empty_response_count": 0,
                    "measurement_empty_response_count": 0,
                },
            }
        else:
            with _working_directory(repo):
                diagnosis, correctness, dialogue, meta = adapter.evaluate_case(
                    scenario, config
                )
        row, conversation = runtime["build_result_row"](
            dataset="MedQA",
            split=str(item["split"]),
            case_id=int(item["scenario_id"]),
            scenario=scenario,
            diagnosis=diagnosis,
            correctness=bool(correctness),
            dialogue=dialogue,
            meta=meta,
            contract_dry_run=bool(payload["contract_dry_run"]),
        )
    except Exception as exc:  # keep one backend failure from killing the run
        row, conversation = _failure_row(item, exc)
    return position, row, conversation, system_prompt, task_prompt


def run_batch(
    *,
    items: list[dict],
    skill_content: str,
    out_root: str,
    agentclinic_repo: str,
    doctor_llm: str,
    patient_llm: str,
    measurement_llm: str,
    moderator_llm: str,
    total_inferences: int,
    contract_dry_run: bool = False,
    workers: int = 1,
) -> list[dict]:
    """Evaluate a ReflACT skill with no AgentClinic harness artifact."""
    output_dir = Path(out_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(agentclinic_repo).expanduser().resolve()
    output_dir.joinpath("skill.md").write_text(
        skill_content.rstrip() + "\n", encoding="utf-8"
    )
    base_payload = {
        "skill_content": skill_content,
        "output_dir": str(output_dir),
        "agentclinic_repo": str(repo),
        "doctor_llm": doctor_llm,
        "patient_llm": patient_llm,
        "measurement_llm": measurement_llm,
        "moderator_llm": moderator_llm,
        "total_inferences": int(total_inferences),
        "contract_dry_run": bool(contract_dry_run),
    }
    payloads = [
        {**base_payload, "position": index, "item": item}
        for index, item in enumerate(items)
    ]
    completed: dict[int, tuple[dict, list[dict], str, str]] = {}
    worker_count = min(max(1, int(workers)), max(len(payloads), 1))
    if worker_count == 1:
        outputs = map(_run_one, payloads)
        for position, row, conversation, system_prompt, task_prompt in outputs:
            completed[position] = (row, conversation, system_prompt, task_prompt)
    else:
        try:
            pool = ProcessPoolExecutor(max_workers=worker_count)
        except (NotImplementedError, PermissionError) as exc:
            print(
                "[skillopt-agentclinic] process pool unavailable; "
                f"falling back to one worker: {exc}"
            )
            outputs = map(_run_one, payloads)
            for position, row, conversation, system_prompt, task_prompt in outputs:
                completed[position] = (
                    row,
                    conversation,
                    system_prompt,
                    task_prompt,
                )
        else:
            with pool:
                futures = {
                    pool.submit(_run_one, payload): payload for payload in payloads
                }
                for future in as_completed(futures):
                    position, row, conversation, system_prompt, task_prompt = (
                        future.result()
                    )
                    completed[position] = (
                        row,
                        conversation,
                        system_prompt,
                        task_prompt,
                    )

    runtime = _load_runtime(repo)
    results: list[dict] = []
    for position, item in enumerate(items):
        row, conversation, system_prompt, task_prompt = completed[position]
        runtime["write_prediction_artifacts"](
            output_dir, str(item["id"]), conversation, system_prompt, task_prompt
        )
        results.append(row)
        print(
            f"[skillopt-agentclinic] {position + 1}/{len(items)} id={item['id']} "
            f"hard={row['hard']} agent_ok={row.get('agent_ok', True)}"
        )

    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    observed_calls = sum(int(row.get("observed_model_calls", 0) or 0) for row in results)
    summary = {
        "n": len(results),
        "hard": sum(float(row["hard"]) for row in results) / max(len(results), 1),
        "soft": sum(float(row["soft"]) for row in results) / max(len(results), 1),
        "agent_failures": sum(row.get("agent_ok") is False for row in results),
        "observed_model_calls": observed_calls,
        "harness": None,
        "measurement_mode": "generative",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if (
        not contract_dry_run
        and results
        and all(row.get("agent_ok") is False for row in results)
    ):
        raise RuntimeError("All AgentClinic rollouts failed before a valid trajectory")
    return results
