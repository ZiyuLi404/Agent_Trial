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
from change_generators.skills import SkillArtifact
from trial.trial_manager import parse_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired AgentClinic comparison with an external skill adapter"
    )
    parser.add_argument("--doctor_llm", default="deepseek-v4-pro")
    parser.add_argument("--patient_llm", default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", default="deepseek-v4-flash")
    parser.add_argument("--moderator_llm", default="deepseek-v4-flash")
    parser.add_argument("--dataset", default="MedQA",
                        choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"])
    parser.add_argument("--cases", required=True,
                        help="Case range '0-19' or comma-separated list '1,3,4'")
    parser.add_argument("--skill_path", required=True)
    parser.add_argument("--total_inferences", type=int, default=20)
    parser.add_argument("--output_dir", default="results/skill_experiments/agentclinic")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def comparison_payload(args, skill, paired_results, counts) -> dict:
    completed = len(paired_results)
    return {
        "comparison": "no_skill_vs_with_skill",
        "agent_system": "agentclinic",
        "doctor_llm": args.doctor_llm,
        "dataset": args.dataset,
        "cases": args.cases,
        "patient_llm": args.patient_llm,
        "measurement_llm": args.measurement_llm,
        "moderator_llm": args.moderator_llm,
        "total_inferences": args.total_inferences,
        "skill": skill.to_dict(),
        "completed_pairs": completed,
        "summary": {
            "no_skill_correct": counts["no_skill"],
            "with_skill_correct": counts["with_skill"],
            "no_skill_accuracy": round(counts["no_skill"] / completed, 4) if completed else 0,
            "with_skill_accuracy": round(counts["with_skill"] / completed, 4) if completed else 0,
            "accuracy_delta": round(
                (counts["with_skill"] - counts["no_skill"]) / completed, 4
            ) if completed else 0,
            "improved_cases": counts["improved"],
            "regressed_cases": counts["regressed"],
        },
        "results": paired_results,
    }


def main() -> None:
    args = build_parser().parse_args()
    skill = SkillArtifact.load(args.skill_path)
    adapters = {
        "no_skill": AgentClinicAdapter(),
        "with_skill": AgentClinicAdapter(skill),
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
        prompt_path = output_dir / "skill_prompt_comparison.txt"
        prompt_path.write_text(
            "=== NO SKILL ===\n"
            + prompts["no_skill"]
            + "\n\n=== WITH SKILL ===\n"
            + prompts["with_skill"]
            + "\n",
            encoding="utf-8",
        )
        print(f"Dry run complete; no API calls were made: {prompt_path}")
        print(f"Skill SHA-256: {skill.sha256}")
        return

    paired_results = []
    counts = {"no_skill": 0, "with_skill": 0, "improved": 0, "regressed": 0}
    out_path = output_dir / "skill_comparison.json"

    for pair_index, (case_id, timestamp, scenario) in enumerate(cases):
        order = ["no_skill", "with_skill"] if pair_index % 2 == 0 else ["with_skill", "no_skill"]
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

        baseline_correct = condition_results["no_skill"]["correct"]
        skill_correct = condition_results["with_skill"]["correct"]
        counts["improved"] += int(not baseline_correct and skill_correct)
        counts["regressed"] += int(baseline_correct and not skill_correct)
        paired_results.append({
            "case_id": case_id,
            "timestamp": timestamp,
            "execution_order": order,
            "correct_diagnosis": str(scenario.diagnosis_information()),
            **condition_results,
        })
        out_path.write_text(
            json.dumps(
                comparison_payload(args, skill, paired_results, counts),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"\nDone: {out_path}")


if __name__ == "__main__":
    main()
