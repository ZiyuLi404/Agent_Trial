"""Resolve and verify the read-only paper code checkouts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
VERSIONS_PATH = Path(__file__).with_name("versions.json")


def versions() -> dict:
    return json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))


def resolve_checkout(name: str) -> Path:
    if name not in versions():
        raise ValueError(f"Unknown upstream project: {name}")
    env_name = name.upper().replace("-", "_") + "_UPSTREAM"
    candidates = []
    if os.environ.get(env_name):
        candidates.append(Path(os.environ[env_name]))
    candidates.extend(
        [
            ROOT / "upstream" / "checkouts" / name,
            PROJECT_ROOT.parent / "related_work" / "code" / name,
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.joinpath(".git").exists():
            return resolved
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No checkout found for {name}. Searched:\n  - {searched}\n"
        "Run: python -m skill_harness.upstream.sync"
    )


def verify_checkout(name: str, *, require_clean: bool = True) -> Path:
    checkout = resolve_checkout(name)
    expected = versions()[name]["commit"]
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"{name} is at {actual}; expected pinned commit {expected}")
    if require_clean:
        status = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain"], text=True
        ).strip()
        if status:
            raise RuntimeError(
                f"{name} checkout is modified. Paper code must remain read-only:\n{status}"
            )
    return checkout


__all__ = ["resolve_checkout", "verify_checkout", "versions"]
