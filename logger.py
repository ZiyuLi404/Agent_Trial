import json

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
