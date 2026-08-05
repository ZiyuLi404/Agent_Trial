"""Clone the two paper repositories at the versions recorded in versions.json."""

from __future__ import annotations

import subprocess
from pathlib import Path

from skill_harness.upstream import ROOT, versions


def main() -> None:
    destination_root = ROOT / "upstream" / "checkouts"
    destination_root.mkdir(parents=True, exist_ok=True)
    for name, spec in versions().items():
        destination = destination_root / name
        if not destination.exists():
            subprocess.run(
                ["git", "clone", spec["url"], str(destination)], check=True
            )
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "origin", spec["commit"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", spec["commit"]],
            check=True,
        )
        print(f"{name}: {spec['commit']}")


if __name__ == "__main__":
    main()
