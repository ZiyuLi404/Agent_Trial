import json
import os

LOG_FILE = "trial_log.jsonl"


def log_case(case_id, timestamp, version_id, arm, control_model,
             treatment_version, treatment_model,
             diagnosis, correct_diagnosis,
             correctness, confidence, compliance, consultation):
    """Append one case result to the JSONL trial log."""
    record = {
        "case_id": case_id,
        "timestamp": timestamp,
        "version_id": version_id,
        "arm": arm,
        "control_model": control_model,
        "treatment_version": treatment_version,
        "treatment_model": treatment_model,
        "diagnosis": diagnosis,
        "correct_diagnosis": correct_diagnosis,
        "correctness": correctness,
        "confidence": confidence,
        "compliance": compliance,
        "consultation": consultation,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def log_deployment_case(record, output_dir="."):
    """Append one deployment-replay case record to deployment_log.jsonl.

    Uses a separate log file so deployment_replay runs don't pollute the
    standard trial_log.jsonl used by accuracy mode.

    Expected keys in record (all optional except case_id):
      case_id, timestamp, epoch_id, version_id, model_name, arm,
      evaluation_type, source_epoch, dataset, diagnosis, correct_diagnosis,
      correctness, transcript_id, transcript_text, run_id, random_seed.
    """
    log_path = os.path.join(output_dir, "deployment_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record
