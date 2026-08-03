import time
from datetime import datetime, timezone

from AgentClinic.agentclinic import (
    ScenarioLoaderMedQA,
    ScenarioLoaderMedQAExtended,
    ScenarioLoaderNEJM,
    ScenarioLoaderNEJMExtended,
    MeasurementAgent,
    PatientAgent,
    DoctorAgent,
    compare_results,
    EMPTY_SENTINEL,
    _deepseek_debug,
)
from trial.randomization import assign_arm

LOADERS = {
    "MedQA": ScenarioLoaderMedQA,
    "MedQA_Ext": ScenarioLoaderMedQAExtended,
    "NEJM": ScenarioLoaderNEJM,
    "NEJM_Ext": ScenarioLoaderNEJMExtended,
}


def run_case(scenario, config, doctor_factory=None):
    """Run one AgentClinic case.

    Returns (diagnosis, correctness, full_dialogue, meta) where meta contains:
      raw_doctor_response_empty, doctor_retry_count, doctor_empty_response,
      backend_error_message, reasoning_content_present, doctor_reasoning_debug.
    """
    doctor_llm = config["doctor_llm"]
    patient_llm = config["patient_llm"]
    measurement_llm = config["measurement_llm"]
    moderator_llm = config["moderator_llm"]
    total_inferences = config.get("total_inferences", 20)

    meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
    patient_agent = PatientAgent(scenario=scenario, backend_str=patient_llm)
    if doctor_factory is None:
        doctor_agent = DoctorAgent(
            scenario=scenario,
            backend_str=doctor_llm,
            max_infs=total_inferences,
        )
    else:
        doctor_agent = doctor_factory(scenario, config)

    pi_dialogue = ""
    doctor_dialogue = ""
    full_dialogue = ""
    diagnosis = None
    correctness = False
    doctor_retry_count = 0
    doctor_empty_response = False
    backend_error_message = ""
    _last_reasoning_present = False
    _last_reasoning_debug = ""

    def _is_empty(text):
        return not text or text.strip() == "" or text == EMPTY_SENTINEL

    for _inf_id in range(total_inferences):
        if _inf_id == total_inferences - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

        # Snapshot doctor state so we can roll back before a retry.
        _hist_snap = doctor_agent.agent_hist
        _infs_snap = doctor_agent.infs

        doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue)

        if _is_empty(doctor_dialogue):
            # Capture reasoning debug from the failed call before rolling back.
            _last_reasoning_present = _deepseek_debug.get("reasoning_content_present", False)
            _last_reasoning_debug = _deepseek_debug.get("doctor_reasoning_debug", "")
            doctor_retry_count += 1
            print(
                f"[WARN] Empty doctor output on turn {_inf_id} (model={doctor_llm}) "
                f"reasoning_present={_last_reasoning_present}, retrying..."
            )
            doctor_agent.agent_hist = _hist_snap
            doctor_agent.infs = _infs_snap
            doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue)
            if _is_empty(doctor_dialogue):
                # Prefer reasoning debug from retry if it adds new info.
                _retry_r = _deepseek_debug.get("doctor_reasoning_debug", "")
                if _retry_r and not _last_reasoning_debug:
                    _last_reasoning_debug = _retry_r
                    _last_reasoning_present = _deepseek_debug.get("reasoning_content_present", False)
                doctor_empty_response = True
                backend_error_message = (
                    f"Doctor returned empty/sentinel after retry on turn {_inf_id} "
                    f"(model={doctor_llm})"
                )
                print(f"[ERROR] {backend_error_message}. Stopping case.")
                doctor_agent.agent_hist = _hist_snap
                doctor_agent.infs = _infs_snap
                break

        full_dialogue += "Doctor: " + doctor_dialogue + "\n"
        print(f"Doctor [{int((_inf_id + 1) / total_inferences * 100)}%]:", doctor_dialogue)

        if "DIAGNOSIS READY" in doctor_dialogue:
            result = compare_results(
                doctor_dialogue, scenario.diagnosis_information(), moderator_llm, None
            )
            correctness = result.startswith("yes")
            diagnosis = doctor_dialogue
            break

        if "REQUEST TEST" in doctor_dialogue:
            pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
            full_dialogue += "Measurement: " + pi_dialogue + "\n"
            print("Measurement:", pi_dialogue)
            patient_agent.add_hist(pi_dialogue)
        else:
            pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
            full_dialogue += "Patient: " + pi_dialogue + "\n"
            print("Patient:", pi_dialogue)
            meas_agent.add_hist(pi_dialogue)

        time.sleep(1.0)

    if doctor_empty_response:
        final_diagnosis = EMPTY_SENTINEL
    else:
        final_diagnosis = diagnosis or doctor_dialogue

    meta = {
        "raw_doctor_response_empty": doctor_retry_count > 0,
        "doctor_retry_count": doctor_retry_count,
        "doctor_empty_response": doctor_empty_response,
        "backend_error_message": backend_error_message,
        "reasoning_content_present": _last_reasoning_present,
        "doctor_reasoning_debug": _last_reasoning_debug,
    }
    return final_diagnosis, correctness, full_dialogue, meta


def parse_cases(spec, dataset):
    """Yield (case_id, timestamp, scenario) from a range or list spec.

    spec examples: '30-129'  →  ids 30..129 inclusive
                   '1,3,4'   →  ids 1, 3, 4
    """
    loader_cls = LOADERS.get(dataset)
    if loader_cls is None:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(LOADERS)}")
    loader = loader_cls()

    if "," in spec:
        ids = [int(x.strip()) for x in spec.split(",")]
    elif "-" in spec:
        start, end = map(int, spec.split("-", 1))
        ids = list(range(start, end + 1))
    else:
        ids = [int(spec.strip())]

    for case_id in ids:
        scenario = loader.get_scenario(id=case_id % loader.num_scenarios)
        timestamp = datetime.now(timezone.utc).isoformat()
        yield case_id, timestamp, scenario


def stream_cases(dataset, num_cases=None, start_id=0):
    """Yield (case_id, timestamp, scenario, arm) sequentially in case order.

    start_id offsets the global case counter so multi-epoch callers (e.g.
    deployment_timeline) can produce unique case_ids across epochs without
    altering scenario loading — the scenario index wraps via modulo.
    """
    loader_cls = LOADERS.get(dataset)
    if loader_cls is None:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(LOADERS)}")

    loader = loader_cls()
    n = min(num_cases if num_cases is not None else loader.num_scenarios, loader.num_scenarios)

    for i in range(n):
        case_id = start_id + i
        scenario = loader.get_scenario(id=case_id % loader.num_scenarios)
        timestamp = datetime.now(timezone.utc).isoformat()
        arm = assign_arm(case_id)
        yield case_id, timestamp, scenario, arm
