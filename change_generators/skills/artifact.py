import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillArtifact:
    """Immutable, versioned skill content produced manually or by an optimizer."""

    skill_id: str
    version: str
    content: str
    sha256: str
    path: Path
    generated_by: str = "unknown"
    parent_version: str | None = None

    @classmethod
    def load(cls, path_str: str | Path) -> "SkillArtifact":
        path = Path(path_str).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Skill file does not exist: {path}")

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Skill file is empty: {path}")

        metadata_path = path.with_suffix(".meta")
        metadata = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        return cls(
            skill_id=metadata.get("skill_id", path.parent.name),
            version=metadata.get("version", path.stem),
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            path=path,
            generated_by=metadata.get("generated_by", "unknown"),
            parent_version=metadata.get("parent_version"),
        )

    def to_dict(self, include_content: bool = True) -> dict:
        payload = {
            "skill_id": self.skill_id,
            "version": self.version,
            "sha256": self.sha256,
            "path": str(self.path),
            "generated_by": self.generated_by,
            "parent_version": self.parent_version,
        }
        if include_content:
            payload["content"] = self.content
        return payload
