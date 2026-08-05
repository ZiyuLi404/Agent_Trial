"""Method-neutral markdown sample export used by coding-agent optimizers."""

import json
import re
from pathlib import Path


def sanitize_id(raw) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw))[:100]


def _truncate(text, limit: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n\n...[truncated {len(value) - limit} chars]"


def result_to_markdown(row: dict) -> tuple[str, str]:
    status = "passed" if float(row.get("hard", 0)) >= 0.5 else "failed"
    sample_id = sanitize_id(row.get("id", "unknown"))
    body = [
        "---",
        f"id: {sample_id}",
        f"status: {status}",
        f"score: {float(row.get('hard', 0))}",
        "env: agentclinic",
        f"split: {row.get('split', 'unknown')}",
        "---",
        "",
        f"# AgentClinic sample {sample_id} — {status.upper()}",
        "",
        "## Input",
        _truncate(row.get("question", ""), 4000),
        "",
        "## Expected",
        _truncate(row.get("correct_text", ""), 1500),
        "",
        "## Agent output",
        _truncate(row.get("predicted_answer", ""), 4000),
        "",
        "## Metrics",
        f"- hard: {row.get('hard', 0)}",
        f"- soft: {row.get('soft', 0.0)}",
        f"- turns: {row.get('n_turns', 0)}",
        f"- tests_requested: {row.get('tests_requested', 0)}",
    ]
    if row.get("fail_reason"):
        body.extend(["", "## Failure reason", _truncate(row["fail_reason"], 2000)])
    trajectory = row.get("trajectory")
    if trajectory:
        if not isinstance(trajectory, str):
            trajectory = json.dumps(trajectory, indent=2, ensure_ascii=False)
        body.extend(
            [
                "",
                "## Trace",
                "<details>",
                "<summary>Full trajectory</summary>",
                "",
                "```text",
                _truncate(trajectory, 10000),
                "```",
                "</details>",
            ]
        )
    return status, "\n".join(body) + "\n"


def export_samples(results: list[dict], workspace: str | Path, limit: int = 0) -> dict:
    root = Path(workspace) / ".skillopt" / "samples"
    failed_dir = root / "failed"
    passed_dir = root / "passed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    passed_dir.mkdir(parents=True, exist_ok=True)
    for directory in (failed_dir, passed_dir):
        for path in directory.glob("*.md"):
            path.unlink()

    counts = {"failed": 0, "passed": 0}
    for written, row in enumerate(results):
        if limit and written >= limit:
            break
        status, markdown = result_to_markdown(row)
        destination = failed_dir if status == "failed" else passed_dir
        destination.joinpath(f"{sanitize_id(row['id'])}.md").write_text(
            markdown, encoding="utf-8"
        )
        counts[status] += 1
    return counts
