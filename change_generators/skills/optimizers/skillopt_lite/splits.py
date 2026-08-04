import random
import tomllib
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")


def load_split_config(path_str: str | Path) -> dict:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Split config does not exist: {path}")
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    ratios = payload.get("ratios", {})
    values = [int(ratios.get(name, 0)) for name in SPLIT_NAMES]
    if any(value <= 0 for value in values):
        raise ValueError("Split ratios train/val/test must all be positive")
    payload["_path"] = str(path)
    return payload


def build_split_manifest(
    dataset: str,
    num_scenarios: int,
    split_config: dict,
) -> dict:
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be positive")

    expected = split_config.get("expected_sizes", {}).get(dataset)
    if expected is not None and int(expected) != num_scenarios:
        raise ValueError(
            f"Dataset {dataset} has {num_scenarios} scenarios; split config expects {expected}"
        )

    seed = int(split_config.get("seed", 42))
    ratios = [int(split_config["ratios"][name]) for name in SPLIT_NAMES]
    total_ratio = sum(ratios)
    train_n = num_scenarios * ratios[0] // total_ratio
    val_n = num_scenarios * ratios[1] // total_ratio

    ids = list(range(num_scenarios))
    random.Random(seed).shuffle(ids)
    split_ids = {
        "train": ids[:train_n],
        "val": ids[train_n:train_n + val_n],
        "test": ids[train_n + val_n:],
    }
    excluded_ids = {}
    configured_exclusions = split_config.get("exclude_ids", {})
    for split_name in SPLIT_NAMES:
        excluded = [int(case_id) for case_id in configured_exclusions.get(split_name, [])]
        unknown = sorted(set(excluded) - set(split_ids[split_name]))
        if unknown:
            raise ValueError(
                f"Excluded ids are not in the {split_name} split: {unknown}"
            )
        excluded_set = set(excluded)
        split_ids[split_name] = [
            case_id for case_id in split_ids[split_name] if case_id not in excluded_set
        ]
        excluded_ids[split_name] = excluded
    return {
        "dataset": dataset,
        "num_scenarios": num_scenarios,
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "source": split_config.get("_path"),
        "protocol": split_config.get("protocol", {}),
        "excluded_ids": excluded_ids,
        "splits": split_ids,
    }


def select_case_ids(
    manifest: dict,
    split: str,
    eval_limit: int = 0,
    sample_seed: int = 42,
) -> list[int]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; choose from {SPLIT_NAMES}")
    ids = list(manifest["splits"][split])
    if eval_limit < 0:
        raise ValueError("eval_limit cannot be negative")
    if eval_limit == 0 or eval_limit >= len(ids):
        return ids
    return random.Random(sample_seed).sample(ids, eval_limit)
