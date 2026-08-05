"""Run the pinned, unmodified Microsoft SkillOpt trainer on AgentClinic."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from skill_harness.upstream import verify_checkout


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "one_epoch.yaml"


def _bootstrap_credentials() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if deepseek_key:
        os.environ.setdefault("OPENAI_COMPATIBLE_API_KEY", deepseek_key)
    os.environ.setdefault("OPENAI_COMPATIBLE_BASE_URL", "https://api.deepseek.com")


def main(argv: list[str] | None = None) -> None:
    upstream = verify_checkout("SkillOpt")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    _bootstrap_credentials()

    from scripts import train as upstream_train
    from skill_harness.methods.skillopt_original.adapter import (
        AgentClinicSkillOptAdapter,
    )

    upstream_train._ENV_REGISTRY["agentclinic"] = AgentClinicSkillOptAdapter
    args = list(argv if argv is not None else sys.argv[1:])
    if "--config" not in args:
        args = ["--config", str(DEFAULT_CONFIG), *args]
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        sys.argv = ["skillopt-agentclinic", *args]
        upstream_train.main()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
