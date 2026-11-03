"""Metrics with explicit event definitions and no scientific-Python dependency."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import mean, pvariance
from typing import Any, Iterable

Row = dict[str, Any]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1) and isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false", "1", "0"}:
        return value.lower() in {"true", "1"}
    raise ValueError(f"Expected a boolean, received {value!r}")


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def normalized_rows(rows: Iterable[Row]) -> list[Row]:
    result = []
    for row in rows:
        item = dict(row)
        item["judge_score"] = _probability(row["judge_score"], "judge_score")
        item["ground_truth_score"] = _probability(row["ground_truth_score"], "ground_truth_score")
        item["judge_binary"] = as_bool(row["judge_binary"])
        item["pass_at_1"] = as_bool(row["pass_at_1"])
        if item["pass_at_1"] != (item["ground_truth_score"] == 1.0):
            raise ValueError("pass_at_1 must mean that all hidden test units passed")
        if "round" in item:
            round_number = int(item["round"])
            if round_number < 0 or float(item["round"]) != round_number:
                raise ValueError("round must be a nonnegative integer")
            item["round"] = round_number
        result.append(item)
    return result


def evaluation_gap(judge_score: float, ground_truth_score: float) -> float:
    return _probability(judge_score, "judge_score") - _probability(
        ground_truth_score, "ground_truth_score"
    )


def reliability_bins(rows: Iterable[Row], n_bins: int = 10) -> list[Row]:
    """Equal-width bins; the final bin includes score 1. Empty bins stay missing."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    groups: dict[int, list[Row]] = defaultdict(list)
    for row in normalized_rows(rows):
        groups[min(n_bins - 1, int(row["judge_score"] * n_bins))].append(row)
    return [
        {
            "bin": index,
            "lower": index / n_bins,
            "upper": (index + 1) / n_bins,
            "count": len(groups[index]),
            "mean_score": mean(r["judge_score"] for r in groups[index]) if groups[index] else None,
            "empirical_accuracy": mean(float(r["pass_at_1"]) for r in groups[index])
            if groups[index]
            else None,
        }
        for index in range(n_bins)
    ]


def _roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranked = sorted(zip(scores, labels, strict=True))
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(scores: list[float], labels: list[bool]) -> float | None:
    """Step-integrated PR-AUC (average precision), with score ties kept together."""
    positives = sum(labels)
    if not positives or positives == len(labels):
        return None
    ranked = sorted(zip(scores, labels, strict=True), reverse=True)
    area, true_positives, seen, index = 0.0, 0, 0, 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        new_positives = sum(label for _, label in ranked[index:end])
        true_positives += new_positives
        seen += end - index
        area += new_positives / positives * true_positives / seen
        index = end
    return area


def _divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def compute_metrics(rows: Iterable[Row], n_bins: int = 10) -> Row:
    """J predicts full correctness; the diagnostic gap uses fractional correctness."""
    data = normalized_rows(rows)
    n = len(data)
    scores = [r["judge_score"] for r in data]
    truth = [r["ground_truth_score"] for r in data]
    labels = [r["pass_at_1"] for r in data]
    predicted = [r["judge_binary"] for r in data]
    tp = sum(p and y for p, y in zip(predicted, labels, strict=True))
    fp = sum(p and not y for p, y in zip(predicted, labels, strict=True))
    tn = sum(not p and not y for p, y in zip(predicted, labels, strict=True))
    fn = sum(not p and y for p, y in zip(predicted, labels, strict=True))
    bins = reliability_bins(data, n_bins)
    ensemble_variances = []
    for row in data:
        members = row.get("ensemble_scores", [])
        if isinstance(members, str):
            members = json.loads(members or "[]")
        if len(members) > 1:
            ensemble_variances.append(pvariance(_probability(s, "ensemble_score") for s in members))
    return {
        "n_rows": n,
        "n_tasks": len({r.get("task_id", str(i)) for i, r in enumerate(data)}),
        "n_correct": tp + fn,
        "n_incorrect": tn + fp,
        "n_approved": tp + fp,
        "mean_judge_score": mean(scores) if n else None,
        "mean_ground_truth": mean(truth) if n else None,
        "mean_gap": mean(s - g for s, g in zip(scores, truth, strict=True)) if n else None,
        "mean_binary_gap": mean(s - int(y) for s, y in zip(scores, labels, strict=True))
        if n
        else None,
        "pass_at_1_rate": _divide(tp + fn, n),
        "false_approval_rate": _divide(fp, fp + tn),
        "false_rejection_rate": _divide(fn, fn + tp),
        "approved_error_fraction": _divide(fp, fp + tp),
        "false_positive_fraction": _divide(fp, n),
        "accuracy": _divide(tp + tn, n),
        "precision": _divide(tp, tp + fp),
        "recall": _divide(tp, tp + fn),
        "f1": _divide(2 * tp, 2 * tp + fp + fn),
        "brier_score": mean(
            (score - int(label)) ** 2 for score, label in zip(scores, labels, strict=True)
        )
        if n
        else None,
        "ece": sum(
            b["count"] / n * abs(b["mean_score"] - b["empirical_accuracy"])
            for b in bins
            if b["count"]
        )
        if n
        else None,
        "roc_auc": _roc_auc(scores, labels),
        "pr_auc": _average_precision(scores, labels),
        "mean_within_ensemble_variance": mean(ensemble_variances) if ensemble_variances else None,
        "score_decision_disagreement": mean(
            float(p != (s >= 0.5)) for p, s in zip(predicted, scores, strict=True)
        )
        if n
        else None,
    }


def trajectory_metrics(rows: Iterable[Row]) -> list[Row]:
    """Endpoint changes; one trajectory is task × condition × proposer replicate."""
    trajectories: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for row in normalized_rows(rows):
        key = str(row["task_id"]), str(row["condition"]), str(row.get("replicate", 0))
        trajectories[key].append(row)
    by_condition: dict[str, list[Row]] = defaultdict(list)
    for (_, condition, _), trajectory in trajectories.items():
        ordered = sorted(trajectory, key=lambda r: r["round"])
        if len({r["round"] for r in ordered}) != len(ordered):
            raise ValueError("Duplicate round in trajectory")
        first, last = ordered[0], ordered[-1]
        dj = last["judge_score"] - first["judge_score"]
        dg = last["ground_truth_score"] - first["ground_truth_score"]
        by_condition[condition].append(
            {
                "delta_judge": dj,
                "delta_ground_truth": dg,
                "delta_gap": dj - dg,
                "judge_up_truth_down": float(dj > 1e-12 and dg < -1e-12),
                "judge_up_truth_unchanged": float(dj > 1e-12 and abs(dg) <= 1e-12),
                "both_improve": float(dj > 1e-12 and dg > 1e-12),
            }
        )
    return [
        {
            "condition": condition,
            "n_trajectories": len(values),
            **{
                f"mean_{key}" if key.startswith("delta") else f"fraction_{key}": mean(
                    v[key] for v in values
                )
                for key in values[0]
            },
        }
        for condition, values in sorted(by_condition.items())
    ]
