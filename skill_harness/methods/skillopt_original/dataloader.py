"""Deterministic AgentClinic case planning for SkillOpt."""
from __future__ import annotations

import json
import random
from pathlib import Path

from skillopt.datasets.base import BaseDataLoader, BatchSpec


_SPLIT_ALIASES = {
    "train": "train",
    "valid_seen": "val",
    "selection": "val",
    "val": "val",
    "valid_unseen": "test",
    "test": "test",
}


class AgentClinicDataLoader(BaseDataLoader):
    """Load explicit case IDs without changing AgentClinic case contents."""

    def __init__(self, manifest_path: str, *, seed: int = 42, limit: int = 0):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.seed = int(seed)
        self.limit = int(limit)
        self.splits: dict[str, list[int]] = {}

    def setup(self, cfg: dict) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"AgentClinic split manifest does not exist: {self.manifest_path}"
            )
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        raw_splits = payload.get("splits", payload)
        for split in ("train", "val", "test"):
            values = [int(value) for value in raw_splits.get(split, [])]
            if not values:
                raise ValueError(f"AgentClinic split {split!r} is empty")
            if len(values) != len(set(values)):
                raise ValueError(f"AgentClinic split {split!r} contains duplicate IDs")
            self.splits[split] = values[: self.limit or None]

        split_sets = [set(self.splits[name]) for name in ("train", "val", "test")]
        if not (
            split_sets[0].isdisjoint(split_sets[1])
            and split_sets[0].isdisjoint(split_sets[2])
            and split_sets[1].isdisjoint(split_sets[2])
        ):
            raise ValueError("AgentClinic train/val/test IDs must be disjoint")

    def get_train_size(self) -> int:
        return len(self.splits["train"])

    @staticmethod
    def _items(split: str, case_ids: list[int]) -> list[dict]:
        return [
            {
                "id": f"MedQA-{case_id:05d}",
                "scenario_id": case_id,
                "split": split,
                "task_type": "medqa",
            }
            for case_id in case_ids
        ]

    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        del kwargs
        pool = list(self.splits["train"])
        count = min(int(batch_size), len(pool))
        selected = pool if count == len(pool) else random.Random(seed).sample(pool, count)
        return BatchSpec(
            phase="train",
            split="train",
            seed=int(seed),
            batch_size=len(selected),
            payload=self._items("train", selected),
            metadata={"manifest": str(self.manifest_path)},
        )

    def build_eval_batch(
        self, env_num: int, split: str, seed: int, **kwargs
    ) -> BatchSpec:
        del kwargs
        canonical = _SPLIT_ALIASES.get(split)
        if canonical is None:
            raise ValueError(f"Unknown AgentClinic split: {split!r}")
        pool = list(self.splits[canonical])
        count = len(pool) if int(env_num) <= 0 else min(int(env_num), len(pool))
        selected = pool if count == len(pool) else random.Random(seed).sample(pool, count)
        return BatchSpec(
            phase="eval",
            split=split,
            seed=int(seed),
            batch_size=len(selected),
            payload=self._items(canonical, selected),
            metadata={"manifest": str(self.manifest_path)},
        )
