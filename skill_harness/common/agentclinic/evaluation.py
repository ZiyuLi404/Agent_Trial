"""Method-neutral AgentClinic result and trajectory serialization."""
from __future__ import annotations

import json
import re
from pathlib import Path

from AgentClinic.agentclinic import extract_diagnosis_text


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


def build_result_row(*, dataset: str, split: str, case_id: int, scenario,
                     diagnosis, correctness: bool, dialogue: str, meta: dict,
                     contract_dry_run: bool) -> tuple[dict, list[dict]]:
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
    patient_calls = sum(message["speaker"] == "Patient" for message in messages)
    measurement_calls = sum(message["speaker"] == "Measurement" for message in messages)
    measurement_meta = meta.get("measurement", {})
    measurement_mode = measurement_meta.get("mode", "generative")
    interaction_backend = meta.get("interaction_backend", {})
    patient_empty = int(interaction_backend.get("patient_empty_response_count", 0) or 0)
    measurement_empty = int(
        interaction_backend.get("measurement_empty_response_count", 0) or 0
    )
    interaction_calls = patient_calls + (
        measurement_calls if measurement_mode == "generative" else 0
    )
    moderator_calls = int("DIAGNOSIS READY" in response)
    observed_model_calls = 0 if contract_dry_run else (
        n_turns + doctor_retry_count + int(doctor_empty_response)
        + interaction_calls + moderator_calls
    )
    if contract_dry_run:
        fail_reason = "contract-dry-run: no model API call was made"
    elif doctor_empty_response:
        fail_reason = meta.get("backend_error_message") or "doctor returned an empty response"
    elif not correctness:
        fail_reason = f"incorrect diagnosis: predicted {predicted!r}; expected {correct_text!r}"
    else:
        fail_reason = ""
    conversation = messages + [{
        "role": "system",
        "speaker": "Evaluator",
        "content": (
            f"[EVALUATION RESULT]\nCorrect: {bool(correctness)}\n"
            f"Predicted diagnosis: {predicted}\nGold diagnosis: {correct_text}"
        ),
    }]
    row = {
        "id": task_id, "scenario_id": case_id, "dataset": dataset,
        "split": split, "task_type": dataset.lower(), "task_description": question,
        "question": question, "correct_text": correct_text,
        "predicted_answer": predicted, "response": response,
        "hard": int(bool(correctness)), "soft": float(bool(correctness)),
        "agent_ok": (
            not doctor_empty_response and patient_empty == 0
            and measurement_empty == 0 and not contract_dry_run
        ),
        "fail_reason": fail_reason, "n_turns": n_turns,
        "tests_requested": tests_requested, "observed_model_calls": observed_model_calls,
        "backend": {
            "raw_doctor_response_empty": bool(meta.get("raw_doctor_response_empty")),
            "doctor_retry_count": doctor_retry_count,
            "doctor_empty_response": doctor_empty_response,
            "backend_error_message": str(meta.get("backend_error_message", "") or ""),
            "reasoning_content_present": bool(meta.get("reasoning_content_present")),
        },
        "measurement": measurement_meta,
        "interaction_backend": {
            "patient_empty_response_count": patient_empty,
            "measurement_empty_response_count": measurement_empty,
        },
        "trajectory": conversation, "variant": meta.get("variant", {}),
        "phase": "contract_dry_run" if contract_dry_run else "evaluation",
    }
    return row, conversation


def write_prediction_artifacts(output_dir: Path, task_id: str,
                               conversation: list[dict], system_prompt: str,
                               task_prompt: str) -> None:
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
