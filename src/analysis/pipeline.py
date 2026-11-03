"""Rebuild inspectable analysis artifacts from saved trajectory records."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analysis.statistics import (
    paired_comparisons,
    paired_endpoint_comparisons,
    summarize_condition,
    summarize_trajectory,
)
from src.metrics.core import Row, normalized_rows, reliability_bins


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_trajectories(rows: list[Row]) -> list[Row]:
    data = normalized_rows(rows)
    if not data:
        raise ValueError("No result rows supplied")
    for field in ("experiment_id", "run_kind"):
        values = {r.get(field) for r in data}
        if len(values) != 1 or None in values or "" in values:
            raise ValueError(f"Analyze one {field} at a time; mixed or missing values are unsafe")
    if data[0].get("run_kind") not in {"mock", "real"}:
        raise ValueError("run_kind must explicitly identify mock or real")
    seen = set()
    condition_keys: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    for row in data:
        key = str(row["task_id"]), str(row.get("replicate", 0)), row["round"]
        observation = str(row["condition"]), *key
        if observation in seen:
            raise ValueError("Duplicate task/condition/replicate/round record")
        seen.add(observation)
        condition_keys[str(row["condition"])].add(key)
    first_keys = next(iter(condition_keys.values()))
    if any(keys != first_keys for keys in condition_keys.values()):
        raise ValueError("Conditions must contain the same complete task/replicate/round set")
    expected_rounds = set(range(max(r["round"] for r in data) + 1))
    trajectory_rounds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for task, replicate, round_number in first_keys:
        trajectory_rounds[task, replicate].add(round_number)
    if any(rounds != expected_rounds for rounds in trajectory_rounds.values()):
        raise ValueError("Every trajectory must include all rounds from 0 to the final round")
    return data


def input_digest(rows: list[Row]) -> str:
    ordered = sorted(
        rows,
        key=lambda r: (
            str(r["task_id"]),
            str(r["condition"]),
            str(r.get("replicate", 0)),
            r["round"],
        ),
    )
    return hashlib.sha256(
        json.dumps(ordered, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def analyze(
    rows: list[Row],
    output_dir: Path,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Final-round comparisons and paired endpoint changes with task-level CIs."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    data = validate_trajectories(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = sorted({str(r["condition"]) for r in data})
    final_round = max(r["round"] for r in data)
    final = [r for r in data if r["round"] == final_round]
    summaries, rounds, trajectories, reliability = [], [], [], []
    for condition in conditions:
        subset = [r for r in data if r["condition"] == condition]
        final_subset = [r for r in subset if r["round"] == final_round]
        summaries.append(
            {
                "condition": condition,
                "round": final_round,
                **summarize_condition(final_subset, bootstrap_samples, seed),
            }
        )
        trajectories.append(summarize_trajectory(subset, bootstrap_samples, seed))
        for round_number in range(final_round + 1):
            round_subset = [r for r in subset if r["round"] == round_number]
            rounds.append(
                {
                    "condition": condition,
                    "round": round_number,
                    **summarize_condition(round_subset, bootstrap_samples, seed),
                }
            )
        reliability.extend(
            {"condition": condition, "round": final_round, **b}
            for b in reliability_bins(final_subset)
        )
    comparisons = paired_comparisons(final, samples=bootstrap_samples, seed=seed)
    endpoint_comparisons = paired_endpoint_comparisons(data, samples=bootstrap_samples, seed=seed)
    provenance = {
        "experiment_id": data[0].get("experiment_id"),
        "run_kind": data[0]["run_kind"],
        "input_sha256": input_digest(data),
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(data),
        "n_tasks": len({r["task_id"] for r in data}),
        "final_round": final_round,
        "conditions": conditions,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "figures_generated": False,
        "confidence_interval": "95% percentile bootstrap, task clusters resampled with replacement; rounds and conditions remain paired",
        "ci_undefined_rule": "CI omitted for fewer than 2 tasks or fewer than 80% defined bootstrap draws; point estimate retained when defined",
        "primary_population": "Final round only; every task/replicate appears in every condition and every round",
        "definitions": {
            "ground_truth": "Fraction of hidden test units passed (not exhaustive semantic correctness)",
            "mean_gap": "Mean of J minus fractional G; a proxy-mismatch diagnostic, not a calibration error",
            "mean_binary_gap": "Signed probability bias: mean of J minus the binary all-hidden-tests-pass event",
            "brier_ece": "J predicts the binary event that all hidden test units pass",
            "false_approval_rate": "P(approve | at least one hidden test unit fails)",
            "false_rejection_rate": "P(reject | all hidden test units pass)",
            "approved_error_fraction": "P(at least one hidden test unit fails | approve)",
            "pr_auc": "Average precision, step-integrated PR area with tied scores grouped; omitted for single-class samples",
            "effect_size": "Paired d_z for task-mean final-gap differences; undefined when paired differences have zero sample variance",
            "mean_within_ensemble_variance": "Mean population variance of member scores on each candidate; a disagreement diagnostic, not the sampling variance of an ensemble estimator or evidence that ensemble variance is reduced",
            "endpoint_comparisons": "Paired condition-minus-reference differences of round-0-to-final changes; references include baseline and baseline_no_feedback when present",
        },
        "limitations": [
            "Task bootstrap captures task sampling variation, not uncertainty across independently trained models.",
            "Final candidates differ across condition-specific optimization trajectories: comparisons estimate the full feedback policy, not judge-only effects.",
            "All comparisons are exploratory; no multiplicity-corrected significance claims are made.",
            "Mock measurements validate plumbing only; synthetic scores are not empirical research findings.",
        ],
    }
    for name, values in (
        ("summary", summaries),
        ("rounds", rounds),
        ("trajectories", trajectories),
        ("paired_comparisons", comparisons),
        ("endpoint_comparisons", endpoint_comparisons),
        ("reliability", reliability),
    ):
        write_csv(output_dir / f"{name}.csv", values)
    result = {
        "provenance": provenance,
        "summary": summaries,
        "rounds": rounds,
        "trajectories": trajectories,
        "paired_comparisons": comparisons,
        "endpoint_comparisons": endpoint_comparisons,
        "reliability": reliability,
    }
    if make_figures:
        from src.analysis.figures import generate_figures

        figure_paths = generate_figures(data, result, output_dir / "figures")
        result["figures"] = [str(path) for path in figure_paths]
        result["figure_files"] = [path.name for path in figure_paths]
        provenance["figures_generated"] = True
    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result
