import argparse
import json
import os
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed; .env will not be auto-loaded. "
          "Run `pip install python-dotenv` or set keys via export/CLI flags.")

from version_manager import get_current_version, open_new_version
from trial_manager import stream_cases, parse_cases, run_case
from logger import log_case

CONTROL_MODEL = "deepseek-v4-flash"


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Clinical Trial — AgentClinic wrapper")
    # Evaluation mode
    parser.add_argument("--eval_mode", default="accuracy",
                        choices=["accuracy", "deployment_replay", "compare"],
                        help="'accuracy' = standard RCT; 'deployment_replay' = multi-epoch shadow eval; "
                             "'compare' = run multiple doctor models on the same cases")
    parser.add_argument("--control_llm", default=CONTROL_MODEL,
                        help="Frozen control doctor model (fixed across all epochs)")
    parser.add_argument("--doctor_llm", default="deepseek-v4-pro",
                        help="Treatment doctor model (accuracy mode only)")
    parser.add_argument("--treatment_schedule", default=None,
                        help="Comma-separated ordered treatment models, one per epoch (deployment_replay only)")
    parser.add_argument("--epoch_sizes", default=None,
                        help="Comma-separated new-case counts per epoch, parallel to --treatment_schedule")
    parser.add_argument("--patient_llm", default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", default="deepseek-v4-flash")
    parser.add_argument("--moderator_llm", default="deepseek-v4-flash")
    parser.add_argument("--dataset", default="MedQA",
                        choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"])
    parser.add_argument("--num_cases", type=int, default=None,
                        help="Number of cases to run (accuracy mode; default: all)")
    parser.add_argument("--cases", default=None,
                        help="Cases to run in compare mode: range '30-129' or list '1,3,4'")
    parser.add_argument("--doctor_llms", default=None,
                        help="Comma-separated doctor models for compare mode, e.g. 'modelA,modelB,modelC,modelD'")
    parser.add_argument("--total_inferences", type=int, default=20,
                        help="Max doctor-patient turns per case")
    parser.add_argument("--output_dir", default="results/deployment_replay",
                        help="Root output directory (deployment_replay mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (deployment_replay mode)")
    # Version management (accuracy mode)
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

    # ── deployment_replay mode ────────────────────────────────────────────────
    if args.eval_mode == "deployment_replay":
        from deployment_timeline import DeploymentTimeline

        if not args.treatment_schedule:
            parser.error("--treatment_schedule is required for deployment_replay mode")
        if not args.epoch_sizes:
            parser.error("--epoch_sizes is required for deployment_replay mode")

        treatment_schedule = [m.strip() for m in args.treatment_schedule.split(",")]
        epoch_sizes = [int(x.strip()) for x in args.epoch_sizes.split(",")]

        if len(treatment_schedule) != len(epoch_sizes):
            parser.error("--treatment_schedule and --epoch_sizes must have the same number of entries")

        timeline = DeploymentTimeline(
            control_model=args.control_llm,
            treatment_schedule=treatment_schedule,
            epoch_sizes=epoch_sizes,
            dataset=args.dataset,
            total_inferences=args.total_inferences,
            output_dir=args.output_dir,
            seed=args.seed,
            patient_llm=args.patient_llm,
            measurement_llm=args.measurement_llm,
            moderator_llm=args.moderator_llm,
        )
        timeline.run()
        return

    # ── compare mode ─────────────────────────────────────────────────────────
    if args.eval_mode == "compare":
        if not args.cases:
            parser.error("--cases is required for compare mode (e.g. '30-129' or '1,3,4')")
        if not args.doctor_llms:
            parser.error("--doctor_llms is required for compare mode (e.g. 'modelA,modelB')")

        doctor_llms = [m.strip() for m in args.doctor_llms.split(",")]
        shared = {
            "patient_llm": args.patient_llm,
            "measurement_llm": args.measurement_llm,
            "moderator_llm": args.moderator_llm,
            "total_inferences": args.total_inferences,
        }

        os.makedirs(args.output_dir, exist_ok=True)

        for doctor_llm in doctor_llms:
            config = {"doctor_llm": doctor_llm, **shared}
            results = []
            correct = 0

            print(f"\n{'=' * 50}")
            print(f"Doctor model: {doctor_llm}")
            print(f"Cases: {args.cases}  |  Dataset: {args.dataset}")
            print(f"{'=' * 50}")

            for case_id, timestamp, scenario in parse_cases(args.cases, args.dataset):
                print(f"\n--- Case {case_id} ---")
                diagnosis, correctness, dialogue, _ = run_case(scenario, config)
                correct += 1 if correctness else 0
                results.append({
                    "case_id": case_id,
                    "timestamp": timestamp,
                    "correct_diagnosis": str(scenario.diagnosis_information()),
                    "output_diagnosis": str(diagnosis),
                    "correct": correctness,
                    "conversation": dialogue,
                })
                print(f"  {'CORRECT' if correctness else 'INCORRECT'} | running accuracy: {correct}/{len(results)}")

            safe_name = re.sub(r"[^\w\-]", "_", doctor_llm)
            out_path = os.path.join(args.output_dir, f"{safe_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "doctor_llm": doctor_llm,
                    "dataset": args.dataset,
                    "cases": args.cases,
                    "patient_llm": args.patient_llm,
                    "measurement_llm": args.measurement_llm,
                    "moderator_llm": args.moderator_llm,
                    "total_cases": len(results),
                    "correct": correct,
                    "accuracy": round(correct / len(results), 4) if results else 0,
                    "results": results,
                }, f, indent=2, ensure_ascii=False)
            print(f"\nSaved: {out_path}  ({correct}/{len(results)} correct)")

        return

    # ── accuracy mode (default) ───────────────────────────────────────────────
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
        diagnosis, correctness, consultation, _ = run_case(scenario, case_config)
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
