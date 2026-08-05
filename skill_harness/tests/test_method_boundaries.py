import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = ROOT / "skill_harness" / "methods"


def test_original_skillopt_has_no_lite_or_harnessopt_dependency():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (METHODS / "skillopt_original").rglob("*.py")
    )
    assert "skillopt_lite" not in source
    assert "harnessopt" not in source


def test_harnessopt_allowlist_is_inside_isolated_workspace():
    payload = json.loads(
        (METHODS / "harnessopt" / "allowlist.json").read_text(encoding="utf-8")
    )
    assert payload["editable"]
    assert all(
        path.startswith("skill_harness/methods/harnessopt/workspace/")
        for path in payload["editable"]
    )
    assert all((ROOT / path).is_file() for path in payload["editable"])


def test_pinned_upstreams_are_clean():
    from skill_harness.upstream import verify_checkout

    assert verify_checkout("SkillOpt").is_dir()
    assert verify_checkout("SkillOpt-Lite").is_dir()


def test_all_tracked_changes_from_refactor_are_sidecar_only():
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "refactor"], cwd=ROOT, text=True
    ).splitlines()
    assert all(path.startswith("skill_harness/") for path in changed)
