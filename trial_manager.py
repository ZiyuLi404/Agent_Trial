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
)
from randomization import assign_arm

LOADERS = {
    "MedQA": ScenarioLoaderMedQA,
    "MedQA_Ext": ScenarioLoaderMedQAExtended,
    "NEJM": ScenarioLoaderNEJM,
    "NEJM_Ext": ScenarioLoaderNEJMExtended,
}


def run_case(scenario, config):
    """Run one AgentClinic case. Returns (diagnosis, correctness, full_dialogue)."""
    doctor_llm = config["doctor_llm"]
    patient_llm = config["patient_llm"]
    measurement_llm = config["measurement_llm"]
    moderator_llm = config["moderator_llm"]
    total_inferences = config.get("total_inferences", 20)

    meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
    patient_agent = PatientAgent(scenario=scenario, backend_str=patient_llm)
    doctor_agent = DoctorAgent(scenario=scenario, backend_str=doctor_llm, max_infs=total_inferences)

    pi_dialogue = ""
    doctor_dialogue = ""
    full_dialogue = ""
    diagnosis = None
    correctness = False

    for _inf_id in range(total_inferences):
        if _inf_id == total_inferences - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

        doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue)
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

    return diagnosis or doctor_dialogue, correctness, full_dialogue


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
