import argparse
import json
from pathlib import Path


def load_results(path_str: str | Path) -> dict[str, dict]:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Results file does not exist: {path}")

    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = str(row.get("id", ""))
        if not task_id:
            raise ValueError(f"Missing result id at {path}:{line_number}")
        if task_id in rows:
            raise ValueError(f"Duplicate result id {task_id!r} in {path}")
        if "hard" not in row:
            raise ValueError(f"Missing hard score for {task_id!r} in {path}")
        rows[task_id] = row
    if not rows:
        raise ValueError(f"Results file is empty: {path}")
    return rows


def compare_result_sets(
    baseline_rows: dict[str, dict],
    candidate_rows: dict[str, dict],
    minimum_delta: float = 0.0,
) -> dict:
    if minimum_delta < 0:
        raise ValueError("minimum_delta cannot be negative")

    baseline_ids = set(baseline_rows)
    candidate_ids = set(candidate_rows)
    if baseline_ids != candidate_ids:
        missing = sorted(baseline_ids - candidate_ids)
        extra = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            f"Paired gate requires identical ids; missing={missing}, extra={extra}"
        )

    case_deltas = []
    for task_id in sorted(baseline_ids):
        baseline = baseline_rows[task_id]
        candidate = candidate_rows[task_id]
        if baseline.get("correct_text") != candidate.get("correct_text"):
            raise ValueError(f"Gold answer changed for paired case {task_id}")
        baseline_hard = float(baseline["hard"])
        candidate_hard = float(candidate["hard"])
        case_deltas.append({
            "id": task_id,
            "correct_text": baseline.get("correct_text"),
            "baseline_hard": baseline_hard,
            "candidate_hard": candidate_hard,
            "delta": candidate_hard - baseline_hard,
            "baseline_answer": baseline.get("predicted_answer", ""),
            "candidate_answer": candidate.get("predicted_answer", ""),
        })

    count = len(case_deltas)
    baseline_hard = sum(row["baseline_hard"] for row in case_deltas) / count
    candidate_hard = sum(row["candidate_hard"] for row in case_deltas) / count
    delta = candidate_hard - baseline_hard
    if delta > minimum_delta:
        action = "promote"
        selected = "candidate"
    elif delta < -minimum_delta:
        action = "reject"
        selected = "baseline"
    else:
        action = "flat"
        selected = "baseline"

    return {
        "n": count,
        "baseline_hard": baseline_hard,
        "candidate_hard": candidate_hard,
        "delta": delta,
        "minimum_delta": minimum_delta,
        "action": action,
        "selected": selected,
        "improved_cases": sum(row["delta"] > 0 for row in case_deltas),
        "regressed_cases": sum(row["delta"] < 0 for row in case_deltas),
        "unchanged_cases": sum(row["delta"] == 0 for row in case_deltas),
        "cases": case_deltas,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired validation gate for skill versions")
    parser.add_argument("--baseline", required=True, help="Baseline results.jsonl")
    parser.add_argument("--candidate", required=True, help="Candidate results.jsonl")
    parser.add_argument("--minimum_delta", type=float, default=0.0)
    parser.add_argument("--output", default=None, help="Optional JSON audit path")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    report = compare_result_sets(
        load_results(args.baseline),
        load_results(args.candidate),
        minimum_delta=args.minimum_delta,
    )
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"Gate: baseline={report['baseline_hard']:.4f} "
        f"candidate={report['candidate_hard']:.4f} "
        f"delta={report['delta']:+.4f} action={report['action']} "
        f"selected={report['selected']}"
    )


if __name__ == "__main__":
    main()
