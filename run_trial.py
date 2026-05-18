import argparse
import os

from version_manager import get_current_version, open_new_version
from trial_manager import stream_cases, run_case
from logger import log_case

CONTROL_MODEL = "deepseek-chat"


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Clinical Trial — AgentClinic wrapper")
    parser.add_argument("--control_llm", default=CONTROL_MODEL,
                        help="Frozen control doctor model (fixed across all epochs)")
    parser.add_argument("--doctor_llm", default="deepseek-v4-pro",
                        help="Treatment doctor model (changes with each version)")
    parser.add_argument("--patient_llm", default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", default="deepseek-v4-flash")
    parser.add_argument("--moderator_llm", default="deepseek-v4-flash")
    parser.add_argument("--dataset", default="MedQA",
                        choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"])
    parser.add_argument("--num_cases", type=int, default=None,
                        help="Number of cases to run (default: all)")
    parser.add_argument("--total_inferences", type=int, default=20,
                        help="Max doctor-patient turns per case")
    # Version management
    parser.add_argument("--new_version", action="store_true",
                        help="Open a new trial version epoch before running")
    parser.add_argument("--version_id", default="v1")
    parser.add_argument("--model_name", default="",
                        help="Human-readable model label for the version record")
    parser.add_argument("--prompt_version", default="p1")
    parser.add_argument("--tool_version", default="t1")
    # API keys
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--deepseek_api_key", default=None)
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key

    if args.new_version:
        version = open_new_version(
            args.version_id,
            args.model_name or args.doctor_llm,
            args.prompt_version,
            args.tool_version,
        )
        print(f"Opened new version: {version}")
    else:
        version = get_current_version()
        print(f"Using version: {version}")

    shared = {
        "patient_llm": args.patient_llm,
        "measurement_llm": args.measurement_llm,
        "moderator_llm": args.moderator_llm,
        "total_inferences": args.total_inferences,
    }
    control_config = {"doctor_llm": args.control_llm, **shared}
    treatment_config = {"doctor_llm": args.doctor_llm, **shared}

    total = 0
    correct = 0

    for case_id, timestamp, scenario, arm in stream_cases(args.dataset, args.num_cases):
        case_config = control_config if arm == "control" else treatment_config
        print(f"\n=== Case {case_id} | arm={arm} | version={version['version_id']} | {timestamp} ===")
        diagnosis, correctness, consultation = run_case(scenario, case_config)
        total += 1
        if correctness:
            correct += 1

        log_case(
            case_id=case_id,
            timestamp=timestamp,
            version_id=version["version_id"],
            arm=arm,
            control_model=args.control_llm,
            treatment_version=version["version_id"],
            treatment_model=args.doctor_llm,
            diagnosis=str(diagnosis),
            correct_diagnosis=str(scenario.diagnosis_information()),
            correctness=correctness,
            confidence=None,
            compliance=None,
            consultation=consultation,
        )

        accuracy = correct / total
        print(f"  Result: {'CORRECT' if correctness else 'INCORRECT'} | Accuracy: {correct}/{total} ({accuracy:.1%})")

    print(f"\n{'=' * 40}")
    print(f"Trial complete: {correct}/{total} ({correct / total:.1%})")
    print(f"Log saved to: trial_log.jsonl")


if __name__ == "__main__":
    main()
