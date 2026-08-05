"""Evaluate the isolated HarnessOpt AgentClinic workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from skill_harness.artifacts import SkillArtifact
from skill_harness.common.agentclinic import LOADERS
from skill_harness.common.agentclinic.evaluation import (
    build_result_row,
    write_prediction_artifacts,
)
from skill_harness.common.sample_export import export_samples
from skill_harness.methods.harnessopt.workspace.agentclinic import (
    HarnessOptAgentClinicAdapter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "skill_harness"
    / "experiments"
    / "agentclinic"
    / "manifests"
    / "medqa_clean_v1.json"
)
DEFAULT_SKILL = (
    PROJECT_ROOT
    / "skill_harness"
    / "artifacts"
    / "seeds"
    / "diagnostic_reasoning"
    / "v000.md"
)


def _code_fingerprint() -> dict:
    allowlist = json.loads(METHOD_ROOT.joinpath("allowlist.json").read_text())
    files = {}
    for relative in allowlist["editable"]:
        path = PROJECT_ROOT / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgentClinic HarnessOpt evaluator")
    parser.add_argument("--skill", default=str(DEFAULT_SKILL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--eval_limit", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doctor_llm", default="deepseek-v4-pro")
    parser.add_argument("--patient_llm", default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", default="deepseek-v4-flash")
    parser.add_argument("--moderator_llm", default="deepseek-v4-flash")
    parser.add_argument("--total_inferences", type=int, default=8)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--contract_dry_run", action="store_true")
    return parser


def evaluate(args: argparse.Namespace) -> dict:
    skill = SkillArtifact.load(args.skill)
    adapter = HarnessOptAgentClinicAdapter(skill_artifact=skill)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    case_ids = list(manifest["splits"][args.split])
    if args.eval_limit and args.eval_limit < len(case_ids):
        case_ids = random.Random(args.seed).sample(case_ids, args.eval_limit)
    loader = LOADERS["MedQA"]()

    workspace = Path(args.workspace).resolve() if args.workspace else (
        PROJECT_ROOT / "skill_harness" / "results" / "harnessopt" / "workspace"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        workspace / ".skillopt" / "_eval_run" / f"{args.split}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "doctor_llm": args.doctor_llm,
        "patient_llm": args.patient_llm,
        "measurement_llm": args.measurement_llm,
        "moderator_llm": args.moderator_llm,
        "total_inferences": args.total_inferences,
    }
    output_dir.joinpath("run_manifest.json").write_text(
        json.dumps(
            {
                "optimizer": "harnessopt",
                "split": args.split,
                "case_ids": case_ids,
                "skill": skill.to_dict(include_content=False),
                "editable_code_sha256": _code_fingerprint(),
                "contract_dry_run": args.contract_dry_run,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = []
    for completed, case_id in enumerate(case_ids, 1):
        scenario = loader.get_scenario(id=int(case_id))
        system_prompt = adapter.build_doctor(scenario, config).system_prompt()
        if args.contract_dry_run:
            diagnosis, correctness, dialogue, meta = "", False, "", {
                "variant": adapter.variant_metadata(),
                "optimizer": "harnessopt",
            }
        else:
            diagnosis, correctness, dialogue, meta = adapter.evaluate_case(
                scenario, config
            )
        row, conversation = build_result_row(
            dataset="MedQA",
            split=args.split,
            case_id=int(case_id),
            scenario=scenario,
            diagnosis=diagnosis,
            correctness=bool(correctness),
            dialogue=dialogue,
            meta=meta,
            contract_dry_run=args.contract_dry_run,
        )
        write_prediction_artifacts(
            output_dir,
            row["id"],
            conversation,
            system_prompt,
            str(scenario.examiner_information()),
        )
        rows.append(row)
        print(
            f"[harnessopt-agentclinic] {completed}/{len(case_ids)} "
            f"id={row['id']} hard={row['hard']}"
        )

    results_path = output_dir / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    samples = export_samples(rows, workspace, args.limit)
    hard = sum(float(row["hard"]) for row in rows) / max(len(rows), 1)
    summary = {
        "n": len(rows),
        "hard": hard,
        "soft": sum(float(row["soft"]) for row in rows) / max(len(rows), 1),
        "samples": samples,
        "results_path": str(results_path),
    }
    output_dir.joinpath("metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results: hard={summary['hard']:.4f} soft={summary['soft']:.4f}")
    return summary


def main(argv=None) -> None:
    evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
