"""Load one immutable split manifest shared by all optimization methods."""

from __future__ import annotations

import json
import random
from pathlib import Path


SPLITS = ("train", "val", "test")


def load_manifest(path_str: str | Path, *, dataset: str, num_scenarios: int) -> dict:
    path = Path(path_str).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset:
        raise ValueError(f"Manifest dataset is {payload.get('dataset')!r}, not {dataset!r}")
    if int(payload.get("num_scenarios", -1)) != int(num_scenarios):
        raise ValueError(
            f"Manifest expects {payload.get('num_scenarios')} scenarios, "
            f"but {dataset} has {num_scenarios}"
        )
    split_sets = []
    for name in SPLITS:
        ids = [int(value) for value in payload.get("splits", {}).get(name, [])]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Manifest split {name!r} is empty or contains duplicates")
        payload["splits"][name] = ids
        split_sets.append(set(ids))
    if not all(
        split_sets[left].isdisjoint(split_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise ValueError("Manifest train/val/test splits overlap")
    excluded = set(int(value) for value in payload.get("excluded_prior_exposure", []))
    formal = set.union(*split_sets)
    if formal & excluded:
        raise ValueError("Excluded prior-exposure IDs appear in a formal split")
    if formal | excluded != set(range(num_scenarios)):
        raise ValueError("Manifest does not account for every scenario ID")
    payload["source"] = str(path)
    return payload


def select_ids(manifest: dict, split: str, limit: int, seed: int) -> list[int]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split!r}")
    ids = list(manifest["splits"][split])
    if limit < 0:
        raise ValueError("eval_limit cannot be negative")
    if limit == 0 or limit >= len(ids):
        return ids
    return random.Random(seed).sample(ids, limit)
