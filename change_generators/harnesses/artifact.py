import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessArtifact:
    """Immutable orchestration/configuration change for an agent system."""

    harness_id: str
    version: str
    agent_system: str
    config: dict
    sha256: str
    path: Path
    generated_by: str = "unknown"
    parent_version: str | None = None

    @classmethod
    def load(cls, path_str: str | Path) -> "HarnessArtifact":
        path = Path(path_str).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Harness file does not exist: {path}")

        raw = path.read_bytes()
        if not raw.strip():
            raise ValueError(f"Harness file is empty: {path}")
        payload = tomllib.loads(raw.decode("utf-8"))
        metadata = payload.get("harness", {})
        config = {key: value for key, value in payload.items() if key != "harness"}

        return cls(
            harness_id=metadata.get("id", path.parent.name),
            version=metadata.get("version", path.stem),
            agent_system=metadata.get("agent_system", "unknown"),
            config=config,
            sha256=hashlib.sha256(raw).hexdigest(),
            path=path,
            generated_by=metadata.get("generated_by", "unknown"),
            parent_version=metadata.get("parent_version"),
        )

    def to_dict(self, include_config: bool = True) -> dict:
        payload = {
            "harness_id": self.harness_id,
            "version": self.version,
            "agent_system": self.agent_system,
            "sha256": self.sha256,
            "path": str(self.path),
            "generated_by": self.generated_by,
            "parent_version": self.parent_version,
        }
        if include_config:
            payload["config"] = self.config
        return payload

    @property
    def doctor_config(self) -> dict:
        doctor = self.config.get("doctor", {})
        if not isinstance(doctor, dict):
            raise ValueError("Harness [doctor] section must be a table")
        return doctor
