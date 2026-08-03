import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent_system_adapters.agentclinic import AgentClinicAdapter
from change_generators.harnesses import HarnessArtifact
from change_generators.skills import SkillArtifact
from trial.trial_manager import parse_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired AgentClinic baseline/variant comparison"
    )
    parser.add_argument("--doctor_llm", default="deepseek-v4-pro")
    parser.add_argument("--patient_llm", default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", default="deepseek-v4-flash")
    parser.add_argument("--moderator_llm", default="deepseek-v4-flash")
    parser.add_argument("--dataset", default="MedQA",
                        choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"])
    parser.add_argument("--cases", required=True,
                        help="Case range '0-19' or comma-separated list '1,3,4'")
    parser.add_argument("--skill_path", default=None,
                        help="Optional versioned skill markdown file")
    parser.add_argument("--harness_path", default=None,
                        help="Optional versioned harness TOML file")
    parser.add_argument("--total_inferences", type=int, default=20,
                        help="Baseline turn budget; a harness may override the variant budget")
    parser.add_argument("--output_dir", default="results/variant_experiments/agentclinic")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def comparison_payload(args, adapter, paired_results, counts) -> dict:
    completed = len(paired_results)
    baseline_config = {
        "total_inferences": args.total_inferences,
    }
    variant_config = adapter.effective_config(baseline_config)
    return {
        "comparison": "baseline_vs_variant",
        "agent_system": "agentclinic",
        "doctor_llm": args.doctor_llm,
        "dataset": args.dataset,
        "cases": args.cases,
        "patient_llm": args.patient_llm,
        "measurement_llm": args.measurement_llm,
        "moderator_llm": args.moderator_llm,
        "baseline_config": baseline_config,
        "variant_config": variant_config,
        "variant": adapter.variant_metadata(),
        "completed_pairs": completed,
        "summary": {
            "baseline_correct": counts["baseline"],
            "variant_correct": counts["variant"],
            "baseline_accuracy": round(counts["baseline"] / completed, 4) if completed else 0,
            "variant_accuracy": round(counts["variant"] / completed, 4) if completed else 0,
            "accuracy_delta": round(
                (counts["variant"] - counts["baseline"]) / completed, 4
            ) if completed else 0,
            "improved_cases": counts["improved"],
            "regressed_cases": counts["regressed"],
        },
        "results": paired_results,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.skill_path and not args.harness_path:
        parser.error("At least one of --skill_path or --harness_path is required")

    skill = SkillArtifact.load(args.skill_path) if args.skill_path else None
    harness = HarnessArtifact.load(args.harness_path) if args.harness_path else None
    adapters = {
        "baseline": AgentClinicAdapter(),
        "variant": AgentClinicAdapter(skill, harness),
    }
    config = {
        "doctor_llm": args.doctor_llm,
        "patient_llm": args.patient_llm,
        "measurement_llm": args.measurement_llm,
        "moderator_llm": args.moderator_llm,
        "total_inferences": args.total_inferences,
    }
    cases = list(parse_cases(args.cases, args.dataset))
    if not cases:
        raise ValueError("No cases selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _, _, scenario = cases[0]
        prompts = {
            name: adapter.build_doctor(scenario, config).system_prompt()
            for name, adapter in adapters.items()
        }
        prompt_path = output_dir / "variant_prompt_comparison.txt"
        prompt_path.write_text(
            "=== BASELINE ===\n"
            + prompts["baseline"]
            + "\n\n=== VARIANT ===\n"
            + prompts["variant"]
            + "\n",
            encoding="utf-8",
        )
        manifest_path = output_dir / "variant_manifest.json"
        manifest_path.write_text(
            json.dumps(adapters["variant"].variant_metadata(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Dry run complete; no API calls were made: {prompt_path}")
        print(f"Variant manifest: {manifest_path}")
        return

    paired_results = []
    counts = {"baseline": 0, "variant": 0, "improved": 0, "regressed": 0}
    out_path = output_dir / "variant_comparison.json"

    for pair_index, (case_id, timestamp, scenario) in enumerate(cases):
        order = ["baseline", "variant"] if pair_index % 2 == 0 else ["variant", "baseline"]
        condition_results = {}
        for condition in order:
            print(f"\n--- Case {case_id}: {condition} ---")
            diagnosis, correctness, dialogue, meta = adapters[condition].evaluate_case(
                scenario, config
            )
            condition_results[condition] = {
                "output_diagnosis": str(diagnosis),
                "correct": bool(correctness),
                "conversation": dialogue,
                "meta": meta,
            }
            counts[condition] += int(bool(correctness))

        baseline_correct = condition_results["baseline"]["correct"]
        variant_correct = condition_results["variant"]["correct"]
        counts["improved"] += int(not baseline_correct and variant_correct)
        counts["regressed"] += int(baseline_correct and not variant_correct)
        paired_results.append({
            "case_id": case_id,
            "timestamp": timestamp,
            "execution_order": order,
            "correct_diagnosis": str(scenario.diagnosis_information()),
            **condition_results,
        })
        out_path.write_text(
            json.dumps(
                comparison_payload(args, adapters["variant"], paired_results, counts),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"\nDone: {out_path}")


if __name__ == "__main__":
    main()
