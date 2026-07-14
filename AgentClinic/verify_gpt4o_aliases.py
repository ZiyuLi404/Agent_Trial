"""Print resolved GPT-4o timepoint snapshot aliases."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AgentClinic.agentclinic import (  # noqa: E402
    GPT4O_SNAPSHOT_ALIASES,
    resolve_openai_model_id,
)


def main() -> None:
    for alias in GPT4O_SNAPSHOT_ALIASES:
        print(f"{alias} -> {resolve_openai_model_id(alias)}")


if __name__ == "__main__":
    main()
