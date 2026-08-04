#!/usr/bin/env python3
"""Algorithm 2: Bayesian pseudo-posterior history borrowing.

Each JSON file in ``--data_dir`` represents one deployed model version. Files
are processed in sorted filename order, as in ``accuracy_summary.py``. When all
versions contain the same patient IDs, the full stream is partitioned into
near-equal contiguous version bundles and assigned in version order. Any
remainder is assigned to the earliest bundles, so every case is used.
Otherwise, each file's complete scored stream is treated as fresh data.

The similarity CSV may be model-level or use the project's replicate labels
(for example, ``flash_1`` and ``flash_2``). Replicate similarities are
collapsed to model level by averaging every valid pair, matching Algorithm 1's
input convention. Borrowing weights are fractional evidence and are never
normalised.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelOutcomes:
    """Ordered fresh-patient rewards for one deployed model version."""

    model: str
    rewards: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.rewards)

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / self.n


def _coerce_reward(value: Any) -> float | None:
    """Convert the correctness representations accepted by accuracy_summary.py."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        reward = float(value)
        return reward if math.isfinite(reward) and 0.0 <= reward <= 1.0 else None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "1", "correct"}:
            return 1.0
        if value in {"false", "no", "0", "incorrect", "wrong"}:
            return 0.0
    return None


def extract_reward(record: dict[str, Any]) -> float | None:
    """Extract a reward from the same flat/nested fields as accuracy_summary.py."""
    fields = ("correct", "is_correct", "correctness", "accuracy", "score")
    for field in fields:
        if field in record:
            reward = _coerce_reward(record[field])
            if reward is not None:
                return reward
    for parent in ("result", "evaluation", "final"):
        nested = record.get(parent)
        if isinstance(nested, dict):
            for field in fields:
                if field in nested:
                    reward = _coerce_reward(nested[field])
                    if reward is not None:
                        return reward
    return None


def _records_from_json(data: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "cases", "data", "items", "records"):
            records = data.get(key)
            if isinstance(records, list):
                return [item for item in records if isinstance(item, dict)]
        # Also support one scored record per JSON file.
        if extract_reward(data) is not None:
            return [data]
    raise ValueError(f"No scored record list found in {path}")


def _load_scored_model_files(
    data_dir: Path,
) -> list[tuple[str, tuple[float, ...], tuple[Any, ...] | None]]:
    """Load every scored model file without selecting its fresh bucket."""
    if not data_dir.is_dir():
        raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")

    loaded: list[tuple[str, tuple[float, ...], tuple[Any, ...] | None]] = []
    for path in sorted(data_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        records = _records_from_json(data, path)
        scored = [
            (record, reward)
            for record in records
            if (reward := extract_reward(record)) is not None
        ]
        rewards = tuple(reward for _, reward in scored)
        if not rewards:
            raise ValueError(f"No valid rewards found in {path}")
        # The filename stem is the canonical project model key used by
        # accuracy_by_25_cases.csv (metadata spelling can differ, e.g. '_' vs '.').
        model = path.stem
        case_ids = (
            tuple(record["case_id"] for record, _ in scored)
            if all("case_id" in record for record, _ in scored)
            else None
        )
        loaded.append((str(model), rewards, case_ids))

    if not loaded:
        raise ValueError(f"No JSON model/version files found in {data_dir}")
    models = [model for model, _, _ in loaded]
    if len(models) != len(set(models)):
        raise ValueError(f"Model names must be unique; found: {models}")
    return loaded


def version_bundle_slices(
    total_cases: int, version_count: int
) -> list[tuple[int, int]]:
    """Partition all cases into near-equal contiguous version bundles.

    Any remainder is assigned one case at a time to the earliest bundles. For
    example, 50 cases across 3 model versions yields sizes 17, 17, and 16.
    """
    if total_cases <= 0 or version_count <= 0:
        raise ValueError("total_cases and version_count must be positive")
    if total_cases < version_count:
        raise ValueError(
            f"Cannot split {total_cases} cases across {version_count} versions"
        )
    base_size, remainder = divmod(total_cases, version_count)
    sizes = [
        base_size + (1 if index < remainder else 0)
        for index in range(version_count)
    ]
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append((start, stop))
        start = stop
    return slices


def load_model_outcome_buckets(data_dir: Path) -> dict[str, list[ModelOutcomes]]:
    """Return every available fresh-patient bucket for every model version."""
    loaded = _load_scored_model_files(data_dir)

    # When every model shares one ordered patient pool, partition the full pool
    # into one near-equal contiguous bundle per deployed version. This consumes
    # every case even when the total is not divisible by the number of models
    # (for example, 50 cases / 3 versions -> 17, 17, 16).
    reference_ids = loaded[0][2]
    shared_patient_pool = (
        reference_ids is not None
        and all(case_ids == reference_ids for _, _, case_ids in loaded)
    )
    buckets_by_model: dict[str, list[ModelOutcomes]] = {}
    if shared_patient_pool:
        bundle_slices = version_bundle_slices(len(reference_ids), len(loaded))
        for model, rewards, _ in loaded:
            buckets_by_model[model] = [
                ModelOutcomes(
                    model=model,
                    rewards=rewards[start:stop],
                )
                for start, stop in bundle_slices
            ]
    else:
        buckets_by_model = {
            model: [ModelOutcomes(model=model, rewards=rewards)]
            for model, rewards, _ in loaded
        }
    return buckets_by_model


def load_full_model_outcomes(data_dir: Path) -> list[ModelOutcomes]:
    """Load every scored case for each model/version, without bucket selection."""
    return [
        ModelOutcomes(model=model, rewards=rewards)
        for model, rewards, _ in _load_scored_model_files(data_dir)
    ]


def load_model_outcomes(data_dir: Path) -> list[ModelOutcomes]:
    """Load the default deployment order and its corresponding fresh buckets."""
    buckets_by_model = load_model_outcome_buckets(data_dir)
    models = list(buckets_by_model)
    return [
        buckets_by_model[model][index]
        if len(buckets_by_model[model]) == len(models)
        else buckets_by_model[model][0]
        for index, model in enumerate(models)
    ]


def load_similarity_matrix(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Load a square CSV whose first column contains row labels."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty similarity CSV: {path}")
        row_label_column = reader.fieldnames[0]
        labels = reader.fieldnames[1:]
        matrix: dict[str, dict[str, float]] = {}
        for row in reader:
            row_label = row[row_label_column].strip()
            matrix[row_label] = {}
            for label in labels:
                try:
                    value = float(row[label])
                except (KeyError, TypeError, ValueError):
                    value = float("nan")
                matrix[row_label][label] = value
    if not labels or not matrix:
        raise ValueError(f"Empty similarity CSV: {path}")
    return labels, matrix


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def infer_replicate_map(models: Iterable[str], labels: list[str]) -> dict[str, list[str]]:
    """Map project model names to direct or ``*_N`` matrix labels."""
    groups: dict[str, list[str]] = {}
    for label in labels:
        base = re.sub(r"[_-]\d+$", "", label)
        groups.setdefault(base, []).append(label)

    result: dict[str, list[str]] = {}
    used_bases: set[str] = set()
    for model in models:
        if model in labels:
            result[model] = [model]
            continue
        model_key = _normalise_name(model)
        candidates = [
            base
            for base in groups
            if model_key == _normalise_name(base)
            or model_key.endswith(_normalise_name(base))
        ]
        if not candidates:
            raise ValueError(
                f"Cannot match model '{model}' to similarity labels. "
                "Use model names or the project's replicate naming convention."
            )
        base = max(candidates, key=lambda item: len(_normalise_name(item)))
        if base in used_bases:
            raise ValueError(f"Similarity label group '{base}' matched multiple models")
        used_bases.add(base)
        result[model] = groups[base]
    return result


def collapse_similarity_matrix(
    models: list[str],
    replicate_map: dict[str, list[str]],
    matrix: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Average replicate-pair scores into model-level s_ij values."""
    collapsed: dict[str, dict[str, float]] = {model: {} for model in models}
    for historical in models:
        for current in models:
            values = [
                matrix[row][column]
                for row in replicate_map[historical]
                for column in replicate_map[current]
                if row in matrix
                and column in matrix[row]
                and math.isfinite(matrix[row][column])
            ]
            if not values:
                raise ValueError(
                    f"No valid similarity values for '{historical}' and '{current}'"
                )
            collapsed[historical][current] = sum(values) / len(values)
    return collapsed


def borrowing_weight(s_ij: float, s_jj: float, lambda_: float) -> float:
    """Compute w_ij = exp[-lambda * max(s_jj - s_ij, 0)]."""
    if lambda_ < 0.0:
        raise ValueError("lambda must be non-negative")
    return math.exp(-lambda_ * max(s_jj - s_ij, 0.0))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction used by the regularized incomplete beta function."""
    max_iterations = 200
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), tiny) * (1.0 if d >= 0.0 else -1.0)
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + coefficient * d
        d = d if abs(d) >= tiny else tiny
        c = 1.0 + coefficient / c
        c = c if abs(c) >= tiny else tiny
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + coefficient * d
        d = d if abs(d) >= tiny else tiny
        c = 1.0 + coefficient / c
        c = c if abs(c) >= tiny else tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("Beta CDF continued fraction did not converge")


def beta_cdf(x: float, alpha: float, beta: float) -> float:
    """CDF of Beta(alpha, beta), implemented without an optional SciPy dependency."""
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_factor = (
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + alpha * math.log(x)
        + beta * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (alpha + 1.0) / (alpha + beta + 2.0):
        return factor * _beta_continued_fraction(alpha, beta, x) / alpha
    return 1.0 - factor * _beta_continued_fraction(beta, alpha, 1.0 - x) / beta


def beta_quantile(probability: float, alpha: float, beta: float) -> float:
    """Numerically invert the Beta CDF by bisection."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if probability in {0.0, 1.0}:
        return probability
    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if beta_cdf(midpoint, alpha, beta) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def beta_credible_interval(
    alpha: float, beta: float, credible_level: float
) -> tuple[float, float]:
    """Return an equal-tailed descriptive interval for the pseudo-posterior."""
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must lie strictly between 0 and 1")
    tail = (1.0 - credible_level) / 2.0
    return (
        beta_quantile(tail, alpha, beta),
        beta_quantile(1.0 - tail, alpha, beta),
    )


def beta_wasserstein_1(
    alpha_a: float,
    beta_a: float,
    alpha_b: float,
    beta_b: float,
    grid_size: int = 1000,
) -> float:
    """Numerically integrate |F_a(u) - F_b(u)| over u in [0, 1]."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    width = 1.0 / grid_size
    return width * sum(
        abs(
            beta_cdf((index + 0.5) * width, alpha_a, beta_a)
            - beta_cdf((index + 0.5) * width, alpha_b, beta_b)
        )
        for index in range(grid_size)
    )


def construct_history_prior(
    current_model: str,
    history: list[ModelOutcomes],
    similarities: dict[str, dict[str, float]],
    lambda_: float,
    alpha0: float,
    beta0: float,
) -> tuple[float, float, list[dict[str, float | str]]]:
    """Construct Beta(alpha_j0, beta_j0) from unnormalised fractional evidence."""
    if alpha0 <= 0.0 or beta0 <= 0.0:
        raise ValueError("alpha0 and beta0 must be positive")

    alpha_j0 = alpha0
    beta_j0 = beta0
    weight_rows: list[dict[str, float | str]] = []
    s_jj = similarities[current_model][current_model]
    for historical in history:
        s_ij = similarities[historical.model][current_model]
        weight = borrowing_weight(s_ij, s_jj, lambda_)
        # alpha_j0 = alpha0 + sum_i w_ij * n_i * r_bar_i
        alpha_evidence = weight * historical.n * historical.mean_reward
        # beta_j0 = beta0 + sum_i w_ij * n_i * (1 - r_bar_i)
        beta_evidence = weight * historical.n * (1.0 - historical.mean_reward)
        alpha_j0 += alpha_evidence
        beta_j0 += beta_evidence
        weight_rows.append(
            {
                "historical_model": historical.model,
                "current_model": current_model,
                "similarity": s_ij,
                "self_similarity": s_jj,
                "weight": weight,
                "historical_n": float(historical.n),
                "historical_mean_reward": historical.mean_reward,
                "borrowed_alpha_evidence": alpha_evidence,
                "borrowed_beta_evidence": beta_evidence,
            }
        )
    return alpha_j0, beta_j0, weight_rows


def online_posterior_trajectory(
    model: str,
    rewards: Iterable[float],
    alpha_j0: float,
    beta_j0: float,
    credible_level: float = 0.95,
) -> list[dict[str, float | int | str]]:
    """Sequentially update the Beta posterior after predicting each outcome."""
    alpha = alpha_j0
    beta = beta_j0
    rows: list[dict[str, float | int | str]] = []
    for patient_index, reward in enumerate(rewards, start=1):
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"Reward must lie in [0, 1], got {reward}")
        prior_alpha = alpha
        prior_beta = beta
        # Prediction is made before the current patient's reward is observed.
        predicted_accuracy = prior_alpha / (prior_alpha + prior_beta)
        predictive_variance = (
            prior_alpha
            * prior_beta
            / ((prior_alpha + prior_beta) ** 2 * (prior_alpha + prior_beta + 1.0))
        )
        predictive_lower, predictive_upper = beta_credible_interval(
            prior_alpha, prior_beta, credible_level
        )
        # Bernoulli/Beta conjugate update: alpha += r_t; beta += 1 - r_t.
        alpha += reward
        beta += 1.0 - reward
        posterior_mean = alpha / (alpha + beta)
        posterior_variance = (
            alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        )
        posterior_lower, posterior_upper = beta_credible_interval(
            alpha, beta, credible_level
        )
        rows.append(
            {
                "model_version": model,
                "patient_index": patient_index,
                "reward": reward,
                "prior_alpha": prior_alpha,
                "prior_beta": prior_beta,
                "predicted_accuracy": predicted_accuracy,
                "predictive_variance": predictive_variance,
                "predictive_interval_lower": predictive_lower,
                "predictive_interval_upper": predictive_upper,
                "alpha": alpha,
                "beta": beta,
                "posterior_mean": posterior_mean,
                "posterior_variance": posterior_variance,
                "posterior_interval_lower": posterior_lower,
                "posterior_interval_upper": posterior_upper,
            }
        )
    return rows


def run_algorithm(
    outcomes: list[ModelOutcomes],
    similarities: dict[str, dict[str, float]],
    lambda_: float,
    alpha0: float,
    beta0: float,
    credible_level: float = 0.95,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run Algorithm 2 for every version in deployment order."""
    trajectory: list[dict[str, Any]] = []
    priors: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    for index, current in enumerate(outcomes):
        alpha_j0, beta_j0, current_weights = construct_history_prior(
            current.model,
            outcomes[:index],
            similarities,
            lambda_,
            alpha0,
            beta0,
        )
        priors.append(
            {
                "model_version": current.model,
                "version_index": index + 1,
                "fresh_patient_count": current.n,
                "alpha_j0": alpha_j0,
                "beta_j0": beta_j0,
                "prior_mean": alpha_j0 / (alpha_j0 + beta_j0),
            }
        )
        weights.extend(current_weights)
        trajectory.extend(
            online_posterior_trajectory(
                current.model,
                current.rewards,
                alpha_j0,
                beta_j0,
                credible_level,
            )
        )
    return trajectory, priors, weights


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def build_all_orders_report(
    buckets_by_model: dict[str, list[ModelOutcomes]],
    similarities: dict[str, dict[str, float]],
    similarity_file: Path,
    lambda_: float,
    alpha0: float,
    beta0: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate Algorithm 2 for every deployment order and fresh-bucket assignment.

    For the project's shared patient pool, deployment step k uses bucket k. This
    matches the layout of the existing ``bayesian_update_embedding_diagnosis``
    reports while retaining Algorithm 2's Beta pseudo-posterior equations.
    """
    models = list(buckets_by_model)
    if any(len(buckets_by_model[model]) not in {1, len(models)} for model in models):
        raise ValueError("All-order reporting requires one bucket or one bucket per model")

    full_posterior = {}
    for model in models:
        full_n = sum(bucket.n for bucket in buckets_by_model[model])
        full_successes = sum(sum(bucket.rewards) for bucket in buckets_by_model[model])
        full_posterior[model] = (
            alpha0 + full_successes,
            beta0 + full_n - full_successes,
            full_n,
            full_successes / full_n,
        )
    similarity_mode = similarity_file.stem.replace("_similarity_matrix", "")
    detail_rows: list[dict[str, Any]] = []

    for order_number, order in enumerate(permutations(models), start=1):
        order_id = f"order_{order_number:02d}"
        model_order = " -> ".join(order)
        deployed: list[ModelOutcomes] = []
        for step_index, target_model in enumerate(order):
            target_buckets = buckets_by_model[target_model]
            target = target_buckets[step_index] if len(target_buckets) > 1 else target_buckets[0]
            alpha_j0, beta_j0, weight_rows = construct_history_prior(
                target_model,
                deployed,
                similarities,
                lambda_,
                alpha0,
                beta0,
            )
            posterior_alpha = alpha_j0 + sum(target.rewards)
            posterior_beta = beta_j0 + target.n - sum(target.rewards)
            prior_mean = alpha_j0 / (alpha_j0 + beta_j0)
            posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
            full_alpha, full_beta, full_n, reference_accuracy = full_posterior[target_model]
            full_posterior_mean = full_alpha / (full_alpha + full_beta)
            signed_error = posterior_mean - full_posterior_mean
            wasserstein_1 = beta_wasserstein_1(
                posterior_alpha,
                posterior_beta,
                full_alpha,
                full_beta,
            )
            similarity_values = [float(row["similarity"]) for row in weight_rows]
            borrowed_n = sum(
                float(row["borrowed_alpha_evidence"])
                + float(row["borrowed_beta_evidence"])
                for row in weight_rows
            )
            historical_n = sum(item.n for item in deployed)
            detail_rows.append(
                {
                    "order_id": order_id,
                    "model_order": model_order,
                    "step_index": step_index,
                    "history_models": " | ".join(item.model for item in deployed),
                    "target_model": target_model,
                    "bucket_id": f"bucket{step_index + 1}" if len(target_buckets) > 1 else "all",
                    "similarity_mode": similarity_mode,
                    "lambda": lambda_,
                    "alpha0": alpha0,
                    "beta0": beta0,
                    "alpha_j0": alpha_j0,
                    "beta_j0": beta_j0,
                    "prior_mean": prior_mean,
                    "current_mean": target.mean_reward,
                    "posterior_alpha": posterior_alpha,
                    "posterior_beta": posterior_beta,
                    "posterior_mean": posterior_mean,
                    "reference_accuracy": reference_accuracy,
                    "full_data_n": full_n,
                    "full_posterior_alpha": full_alpha,
                    "full_posterior_beta": full_beta,
                    "full_posterior_mean": full_posterior_mean,
                    "posterior_mean_signed_error": signed_error,
                    "posterior_mean_absolute_error": abs(signed_error),
                    "wasserstein_1": wasserstein_1,
                    "effective_prior_n": borrowed_n,
                    "current_n": target.n,
                    "historical_n": historical_n,
                    "n_history_models": len(deployed),
                    "similarity_mean": _mean(similarity_values) if similarity_values else "",
                    "similarity_min": min(similarity_values) if similarity_values else "",
                    "similarity_max": max(similarity_values) if similarity_values else "",
                    "metric": "correctness",
                    "input_source": "raw_json",
                }
            )
            deployed.append(target)

    summary_by_order: list[dict[str, Any]] = []
    for order_id in dict.fromkeys(row["order_id"] for row in detail_rows):
        rows = [row for row in detail_rows if row["order_id"] == order_id]
        summary_by_order.append(
            {
                "order_id": order_id,
                "model_order": rows[0]["model_order"],
                "similarity_mode": similarity_mode,
                "lambda": lambda_,
                "alpha0": alpha0,
                "beta0": beta0,
                "posterior_mean_mae": _mean(
                    float(row["posterior_mean_absolute_error"]) for row in rows
                ),
                "mean_wasserstein_1": _mean(float(row["wasserstein_1"]) for row in rows),
                "final_step_posterior_mean_absolute_error": float(
                    rows[-1]["posterior_mean_absolute_error"]
                ),
                "final_step_wasserstein_1": float(rows[-1]["wasserstein_1"]),
                "mean_effective_prior_n": _mean(float(row["effective_prior_n"]) for row in rows),
                "mean_current_n": _mean(float(row["current_n"]) for row in rows),
                "n_rows": len(rows),
                "n_steps": len({row["step_index"] for row in rows}),
            }
        )

    summary_by_model: list[dict[str, Any]] = []
    for model in models:
        rows = [row for row in detail_rows if row["target_model"] == model]
        summary_by_model.append(
            {
                "target_model": model,
                "similarity_mode": similarity_mode,
                "lambda": lambda_,
                "alpha0": alpha0,
                "beta0": beta0,
                "posterior_mean_mae": _mean(
                    float(row["posterior_mean_absolute_error"]) for row in rows
                ),
                "mean_posterior_mean_signed_error": _mean(
                    float(row["posterior_mean_signed_error"]) for row in rows
                ),
                "mean_wasserstein_1": _mean(float(row["wasserstein_1"]) for row in rows),
                "mean_effective_prior_n": _mean(float(row["effective_prior_n"]) for row in rows),
                "mean_current_n": _mean(float(row["current_n"]) for row in rows),
                "mean_step_index": _mean(float(row["step_index"]) for row in rows),
                "n_rows": len(rows),
                "n_orders": len({row["order_id"] for row in rows}),
            }
        )

    best_orders = []
    for rank, row in enumerate(
        sorted(summary_by_order, key=lambda item: item["mean_wasserstein_1"]),
        start=1,
    ):
        best_orders.append({**row, "rank": rank})
    return detail_rows, summary_by_model, summary_by_order, best_orders


def write_all_orders_report(
    output_dir: Path,
    report: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """Write all-order reports using the paper's full-posterior metrics."""
    detail, by_model, by_order, best_orders = report
    write_csv(output_dir / "bayesian_update_all_orders.csv", detail, list(detail[0]))
    write_csv(
        output_dir / "bayesian_update_summary_by_model.csv",
        by_model,
        list(by_model[0]),
    )
    write_csv(
        output_dir / "bayesian_update_summary_by_order.csv",
        by_order,
        list(by_order[0]),
    )
    write_csv(
        output_dir / "best_orders_by_wasserstein.csv",
        best_orders,
        list(best_orders[0]),
    )
    best_by_mean = [
        {**row, "rank": rank}
        for rank, row in enumerate(
            sorted(by_order, key=lambda item: item["posterior_mean_mae"]),
            start=1,
        )
    ]
    write_csv(
        output_dir / "best_orders_by_mae.csv",
        best_by_mean,
        list(best_by_mean[0]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithm 2: Bayesian pseudo-posterior history borrowing"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing one scored JSON result file per model/version.",
    )
    parser.add_argument(
        "--similarity_file",
        required=True,
        help="Model- or replicate-level similarity matrix CSV.",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=10.0,
        help="Weight concentration parameter (default: 10).",
    )
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument(
        "--credible_level",
        type=float,
        default=0.95,
        help="Equal-tailed pseudo-posterior interval level (default: 0.95).",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--all_orders",
        action="store_true",
        help=(
            "Also evaluate every model deployment order against the full-data "
            "Beta reference posterior."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lambda_ < 0.0:
        raise SystemExit("ERROR: --lambda must be non-negative")
    if args.alpha0 <= 0.0 or args.beta0 <= 0.0:
        raise SystemExit("ERROR: --alpha0 and --beta0 must be positive")
    if not 0.0 < args.credible_level < 1.0:
        raise SystemExit("ERROR: --credible_level must lie strictly between 0 and 1")

    try:
        outcomes = load_model_outcomes(Path(args.data_dir))
        labels, raw_similarity = load_similarity_matrix(Path(args.similarity_file))
        models = [item.model for item in outcomes]
        replicate_map = infer_replicate_map(models, labels)
        similarities = collapse_similarity_matrix(models, replicate_map, raw_similarity)
        trajectory, priors, weights = run_algorithm(
            outcomes,
            similarities,
            args.lambda_,
            args.alpha0,
            args.beta0,
            args.credible_level,
        )
        all_orders_report = (
            build_all_orders_report(
                load_model_outcome_buckets(Path(args.data_dir)),
                similarities,
                Path(args.similarity_file),
                args.lambda_,
                args.alpha0,
                args.beta0,
            )
            if args.all_orders
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: {error}") from error

    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "bayesian_trajectory.csv",
        trajectory,
        [
            "model_version", "patient_index", "reward", "prior_alpha", "prior_beta",
            "predicted_accuracy", "predictive_variance", "predictive_interval_lower",
            "predictive_interval_upper", "alpha", "beta", "posterior_mean",
            "posterior_variance", "posterior_interval_lower", "posterior_interval_upper",
        ],
    )
    write_csv(
        output_dir / "history_informed_priors.csv",
        priors,
        [
            "model_version", "version_index", "fresh_patient_count", "alpha_j0",
            "beta_j0", "prior_mean",
        ],
    )
    write_csv(
        output_dir / "borrowing_weights.csv",
        weights,
        [
            "historical_model", "current_model", "similarity", "self_similarity",
            "weight", "historical_n", "historical_mean_reward",
            "borrowed_alpha_evidence", "borrowed_beta_evidence",
        ],
    )
    if all_orders_report is not None:
        write_all_orders_report(output_dir, all_orders_report)
    print(f"Processed {len(outcomes)} model versions in deployment order:")
    print("  " + " -> ".join(item.model for item in outcomes))
    print(f"Saved Bayesian trajectory and prior audit files to: {output_dir}")
    if all_orders_report is not None:
        print("Saved all-order full-posterior Wasserstein and mean-error reports.")


if __name__ == "__main__":
    main()
