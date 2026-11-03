"""Task-cluster percentile bootstrap and explicitly paired comparisons."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, stdev
from typing import Callable, Iterator

from src.metrics.core import Row, compute_metrics, normalized_rows, trajectory_metrics

SUMMARY_METRICS = (
    "mean_judge_score",
    "mean_ground_truth",
    "mean_gap",
    "mean_binary_gap",
    "false_approval_rate",
    "false_rejection_rate",
    "brier_score",
    "ece",
    "pass_at_1_rate",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "approved_error_fraction",
    "false_positive_fraction",
    "mean_within_ensemble_variance",
)


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("A nonempty sample and probability in [0, 1] are required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def task_cluster_samples(rows: list[Row], samples: int, seed: int) -> Iterator[list[Row]]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    clusters: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        clusters[str(row["task_id"])].append(row)
    task_ids = sorted(clusters)
    rng = random.Random(seed)
    for _ in range(samples):
        # Relabel repeated draws so complete trajectories remain separate clusters.
        yield [
            {**row, "task_id": f"bootstrap-{draw}:{task_id}"}
            for draw, task_id in enumerate(rng.choices(task_ids, k=len(task_ids)))
            for row in clusters[task_id]
        ]


def bootstrap_estimates(
    rows: list[Row],
    statistic: Callable[[list[Row]], Row],
    metric_names: tuple[str, ...],
    samples: int = 1000,
    seed: int = 42,
) -> Row:
    """CIs are unavailable for one task or when >20% of replicates are undefined."""
    point = statistic(rows)
    n_tasks = len({str(row["task_id"]) for row in rows})
    estimates: dict[str, list[float]] = {metric: [] for metric in metric_names}
    if n_tasks >= 2:
        for sample in task_cluster_samples(rows, samples, seed):
            values = statistic(sample)
            for metric in metric_names:
                value = values.get(metric)
                if value is not None and math.isfinite(value):
                    estimates[metric].append(float(value))
    result = dict(point)
    minimum_valid = max(2, math.ceil(0.8 * samples))
    for metric in metric_names:
        valid = estimates[metric]
        available = point.get(metric) is not None and len(valid) >= minimum_valid
        result[f"{metric}_ci_low"] = percentile(valid, 0.025) if available else None
        result[f"{metric}_ci_high"] = percentile(valid, 0.975) if available else None
        result[f"{metric}_bootstrap_valid"] = len(valid)
    return result


def summarize_condition(rows: list[Row], samples: int = 1000, seed: int = 42) -> Row:
    return bootstrap_estimates(rows, compute_metrics, SUMMARY_METRICS, samples, seed)


def summarize_trajectory(rows: list[Row], samples: int = 1000, seed: int = 42) -> Row:
    names = (
        "mean_delta_judge",
        "mean_delta_ground_truth",
        "mean_delta_gap",
        "fraction_judge_up_truth_down",
        "fraction_judge_up_truth_unchanged",
        "fraction_both_improve",
    )
    return bootstrap_estimates(
        rows, lambda values: trajectory_metrics(values)[0], names, samples, seed
    )


def paired_comparisons(
    rows: list[Row], reference: str = "baseline", samples: int = 1000, seed: int = 42
) -> list[Row]:
    """Condition minus reference on exactly matched task/replicate observations."""
    data = normalized_rows(rows)
    conditions = sorted({r["condition"] for r in data})
    if reference not in conditions:
        return []

    def keys(row: Row) -> tuple[str, str]:
        return str(row["task_id"]), str(row.get("replicate", 0))

    ref = {keys(r): r for r in data if r["condition"] == reference}
    metrics = (
        "mean_judge_score",
        "mean_ground_truth",
        "mean_gap",
        "false_approval_rate",
        "brier_score",
        "ece",
        "accuracy",
    )
    result = []
    for condition in conditions:
        if condition == reference:
            continue
        treatment = {keys(r): r for r in data if r["condition"] == condition}
        if set(treatment) != set(ref):
            raise ValueError("Paired comparison requires identical task/replicate sets")
        paired_rows = [row for row in data if row["condition"] in {reference, condition}]

        def differences(sample: list[Row], condition_name: str = condition) -> Row:
            base_values = compute_metrics(r for r in sample if r["condition"] == reference)
            condition_values = compute_metrics(
                r for r in sample if r["condition"] == condition_name
            )
            return {
                metric: condition_values[metric] - base_values[metric]
                if condition_values[metric] is not None and base_values[metric] is not None
                else None
                for metric in metrics
            }

        estimates = bootstrap_estimates(paired_rows, differences, metrics, samples, seed)
        task_deltas: dict[str, list[float]] = defaultdict(list)
        for key, row in treatment.items():
            gap = row["judge_score"] - row["ground_truth_score"]
            reference_gap = ref[key]["judge_score"] - ref[key]["ground_truth_score"]
            task_deltas[key[0]].append(gap - reference_gap)
        deltas = [mean(values) for values in task_deltas.values()]
        sd = stdev(deltas) if len(deltas) > 1 else 0.0
        effect_size = mean(deltas) / sd if sd > 1e-12 else None
        for metric in metrics:
            result.append(
                {
                    "condition": condition,
                    "reference": reference,
                    "metric": metric,
                    "estimate": estimates[metric],
                    "ci_low": estimates[f"{metric}_ci_low"],
                    "ci_high": estimates[f"{metric}_ci_high"],
                    "bootstrap_valid": estimates[f"{metric}_bootstrap_valid"],
                    "n_paired_tasks": len(task_deltas),
                    "paired_standardized_effect_dz": effect_size if metric == "mean_gap" else None,
                }
            )
    return result


def paired_endpoint_comparisons(rows: list[Row], samples: int = 1000, seed: int = 42) -> list[Row]:
    """Paired differences of changes, including the no-feedback control when present."""
    data = normalized_rows(rows)
    conditions = sorted({str(r["condition"]) for r in data})
    references = [
        condition for condition in ("baseline", "baseline_no_feedback") if condition in conditions
    ]
    trajectories: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in data:
        trajectories[
            str(row["task_id"]), str(row.get("replicate", 0)), str(row["condition"])
        ].append(row)
    endpoints: list[Row] = []
    for (task, replicate, condition), trajectory in trajectories.items():
        ordered = sorted(trajectory, key=lambda row: row["round"])
        first, last = ordered[0], ordered[-1]
        dj = last["judge_score"] - first["judge_score"]
        dg = last["ground_truth_score"] - first["ground_truth_score"]
        endpoints.append(
            {
                "task_id": task,
                "replicate": replicate,
                "condition": condition,
                "delta_judge": dj,
                "delta_ground_truth": dg,
                "delta_gap": dj - dg,
            }
        )
    names = ("delta_judge", "delta_ground_truth", "delta_gap")
    output = []
    for reference in references:
        for condition in conditions:
            if condition == reference:
                continue
            selected = [row for row in endpoints if row["condition"] in {reference, condition}]

            def differences(
                sample: list[Row], condition_name: str = condition, reference_name: str = reference
            ) -> Row:
                return {
                    metric: mean(
                        row[metric] for row in sample if row["condition"] == condition_name
                    )
                    - mean(row[metric] for row in sample if row["condition"] == reference_name)
                    for metric in names
                }

            estimates = bootstrap_estimates(selected, differences, names, samples, seed)
            condition_values = {
                (r["task_id"], r["replicate"]): r for r in selected if r["condition"] == condition
            }
            reference_values = {
                (r["task_id"], r["replicate"]): r for r in selected if r["condition"] == reference
            }
            if set(condition_values) != set(reference_values):
                raise ValueError("Endpoint comparisons require matched task/replicate sets")
            task_deltas: dict[str, list[float]] = defaultdict(list)
            for key, row in condition_values.items():
                task_deltas[key[0]].append(row["delta_gap"] - reference_values[key]["delta_gap"])
            values = [mean(deltas) for deltas in task_deltas.values()]
            sd = stdev(values) if len(values) > 1 else 0.0
            effect = mean(values) / sd if sd > 1e-12 else None
            for metric in names:
                output.append(
                    {
                        "condition": condition,
                        "reference": reference,
                        "metric": metric,
                        "estimate": estimates[metric],
                        "ci_low": estimates[f"{metric}_ci_low"],
                        "ci_high": estimates[f"{metric}_ci_high"],
                        "bootstrap_valid": estimates[f"{metric}_bootstrap_valid"],
                        "n_paired_tasks": len(task_deltas),
                        "paired_standardized_effect_dz": effect if metric == "delta_gap" else None,
                    }
                )
    return output
