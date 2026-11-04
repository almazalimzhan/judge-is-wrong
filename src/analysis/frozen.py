"""Paired evaluator-quality analysis on identical frozen candidates."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.analysis.pipeline import input_digest, write_csv
from src.analysis.statistics import paired_comparisons, summarize_condition
from src.experiments.frozen import FROZEN_CONDITIONS, _candidate_id
from src.metrics.core import normalized_rows, reliability_bins
from src.utils.serialization import write_json


def validate_frozen(rows: list[dict]) -> list[dict]:
    if any(
        isinstance(row.get("replicate"), bool)
        or not isinstance(row.get("replicate"), int)
        or row["replicate"] < 0
        for row in rows
    ):
        raise ValueError("Frozen replicate identifiers must be nonnegative integers")
    data = normalized_rows(rows)
    if not data or {row.get("experiment_type") for row in data} != {"frozen_candidates"}:
        raise ValueError("Frozen analysis requires frozen_candidates results only")
    for field in ("experiment_id", "run_kind", "corpus_sha256", "ground_truth_protocol"):
        values = {row.get(field) for row in data}
        if len(values) != 1 or None in values or "" in values:
            raise ValueError(f"Frozen analysis requires one explicit {field}")
    if data[0]["run_kind"] not in {"mock", "real"}:
        raise ValueError("Frozen results must identify mock or real execution")
    if re.fullmatch(r"[0-9a-f]{64}", data[0]["corpus_sha256"]) is None:
        raise ValueError("Frozen results need an explicit corpus SHA256")
    if {row["condition"] for row in data} != FROZEN_CONDITIONS:
        raise ValueError("Frozen analysis requires all four evaluator conditions")
    by_candidate: dict[tuple[str, int], list[dict]] = defaultdict(list)
    replicates: dict[str, set[int]] = defaultdict(set)
    for row in data:
        if row["round"] != 0 or row.get("source_round") not in {0, "0"}:
            raise ValueError("Frozen evaluation cannot contain revision rounds")
        original_replicate = row["replicate"]
        if (
            isinstance(original_replicate, bool)
            or not isinstance(original_replicate, int)
            or original_replicate < 0
        ):
            raise ValueError("Frozen replicate identifiers must be nonnegative integers")
        replicate = original_replicate
        if (
            not isinstance(row.get("public_task_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", row["public_task_sha256"]) is None
        ):
            raise ValueError("Frozen result lacks a public-task binding")
        by_candidate[str(row["task_id"]), replicate].append(row)
        replicates[str(row["task_id"])].add(replicate)
        fingerprint = hashlib.sha256(row["candidate_code"].encode()).hexdigest()
        if fingerprint != row.get("candidate_sha256") or row.get("candidate_id") != _candidate_id(
            str(row["task_id"]), replicate, fingerprint
        ):
            raise ValueError("Frozen candidate code does not match its committed identity")
        if row.get("ground_truth_valid") is not True:
            raise ValueError("Frozen comparison needs valid ground-truth execution")
    replicate_sets = list(replicates.values())
    if any(values != replicate_sets[0] for values in replicate_sets):
        raise ValueError("Every frozen task must have the same replicate set")
    fields = (
        "candidate_id",
        "candidate_sha256",
        "candidate_code",
        "problem",
        "public_task_sha256",
        "ground_truth_score",
        "pass_at_1",
        "replicate_seed",
        "public_test_result",
        "hidden_test_result",
    )
    for group in by_candidate.values():
        if (
            len(group) != len(FROZEN_CONDITIONS)
            or {r["condition"] for r in group} != FROZEN_CONDITIONS
        ):
            raise ValueError("Each frozen candidate needs exactly one judgment per condition")
        for field in fields:
            if any(row.get(field) != group[0].get(field) for row in group):
                raise ValueError(f"Evaluator conditions did not receive the same frozen {field}")
    return data


def analyze_frozen(
    rows: list[dict],
    output_dir: Path,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Condition-minus-baseline contrasts, resampling whole task clusters."""
    del make_figures  # Frozen analysis deliberately has no iterative trajectory figures.
    data = validate_frozen(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = sorted(FROZEN_CONDITIONS)
    summary = [
        {
            "condition": condition,
            **summarize_condition(
                [row for row in data if row["condition"] == condition], bootstrap_samples, seed
            ),
        }
        for condition in conditions
    ]
    comparisons = paired_comparisons(data, samples=bootstrap_samples, seed=seed)
    reliability = [
        {"condition": condition, **entry}
        for condition in conditions
        for entry in reliability_bins([row for row in data if row["condition"] == condition])
    ]
    result = {
        "provenance": {
            "experiment_id": data[0]["experiment_id"],
            "experiment_type": "frozen_candidates",
            "run_kind": data[0]["run_kind"],
            "input_sha256": input_digest(data),
            "corpus_sha256": data[0]["corpus_sha256"],
            "ground_truth_protocol": data[0]["ground_truth_protocol"],
            "n_tasks": len({row["task_id"] for row in data}),
            "n_candidates": len(data) // len(conditions),
            "n_rows": len(data),
            "replicates": sorted({int(row["replicate"]) for row in data}),
            "conditions": conditions,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "confidence_interval": "95% percentile task-cluster bootstrap, retaining all candidates/replicates and conditions in each sampled task",
            "estimand": "Evaluator quality on this fixed, equally weighted task-by-replicate candidate corpus; no proposer revision effects",
            "binary_prediction_error": "P(binary approval differs from the all-hidden-tests-pass event), equal to 1 - accuracy",
            "calibration": "Brier score, 10-bin ECE, reliability bins and signed mean J minus binary all-tests-pass event",
            "false_approval_rate": "P(approve | at least one hidden test fails)",
            "paired_direction": "condition minus baseline; lower is better for Brier, ECE, false approval and binary prediction error",
            "limitations": [
                "Task clustering prevents treating proposer replicates as independent tasks.",
                "The selected frozen corpus limits generalization; this is not a causal analysis of feedback-driven trajectories.",
                "ECE is bin-dependent and not additive per candidate; differences are recomputed within each paired bootstrap draw.",
                "Sparse tests and benchmark contamination remain possible; valid execution is not exhaustive correctness.",
                "Intervals are exploratory and unadjusted for multiple comparisons; mock measurements are software validation only.",
            ],
        },
        "summary": summary,
        "paired_comparisons": comparisons,
        "reliability": reliability,
    }
    for name, values in (
        ("summary", summary),
        ("paired_comparisons", comparisons),
        ("reliability", reliability),
    ):
        write_csv(output_dir / f"{name}.csv", values)
    write_json(output_dir / "frozen_analysis.json", result)
    return result


def render_frozen_report(rows: list[dict], analysis_dir: Path, output_path: Path) -> Path:
    data = validate_frozen(rows)
    result = json.loads((Path(analysis_dir) / "frozen_analysis.json").read_text())
    provenance = result["provenance"]
    if provenance["input_sha256"] != input_digest(data):
        raise ValueError("Frozen analysis provenance differs from requested result rows")

    def formatted(row: dict, metric: str, *, paired: bool = False) -> str:
        value = row.get("estimate" if paired else metric)
        low = row.get("ci_low" if paired else f"{metric}_ci_low")
        high = row.get("ci_high" if paired else f"{metric}_ci_high")
        if value is None:
            return "undefined"
        return (
            f"{value:.3f} [{low:.3f}, {high:.3f}]"
            if low is not None
            else f"{value:.3f} [CI unavailable]"
        )

    metrics = (
        "brier_score",
        "false_approval_rate",
        "ece",
        "binary_prediction_error",
        "mean_binary_gap",
    )
    lines = [
        "# Frozen-candidate evaluator comparison",
        "",
        "**Empirical results: PENDING. This mock run validates software only.**"
        if provenance["run_kind"] == "mock"
        else "These descriptive measurements concern this fixed corpus and configured evaluators; no hypothesis confirmation is claimed.",
        "",
        f"Experiment: `{provenance['experiment_id']}`. Corpus SHA256: `{provenance['corpus_sha256']}`.",
        f"Tasks: {provenance['n_tasks']}; candidates: {provenance['n_candidates']}; proposer replicates per task: {len(provenance['replicates'])}.",
        "",
        "Every condition scores exactly the same candidate code and ground-truth outcomes. There are no proposer calls or revisions. This estimates evaluator quality on the frozen corpus, separately from feedback effects on iterative trajectories.",
        "",
        "## Evaluator-quality estimates",
        "",
        "Values show estimate [95% task-cluster bootstrap interval]. Undefined denominators remain missing.",
        "",
        "| Condition | Brier | False approval | ECE | Binary error | Signed binary gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| "
        + row["condition"]
        + " | "
        + " | ".join(formatted(row, metric) for metric in metrics)
        + " |"
        for row in result["summary"]
    )
    lines.extend(
        [
            "",
            "## Paired differences from baseline",
            "",
            "Condition minus baseline. Negative differences favor the condition for Brier, false approval, ECE and binary error; signed gap should be interpreted relative to zero.",
            "",
            "| Condition | Metric | Paired difference [95% CI] |",
            "|---|---|---:|",
        ]
    )
    lines.extend(
        f"| {row['condition']} | {row['metric']} | {formatted(row, row['metric'], paired=True)} |"
        for row in result["paired_comparisons"]
        if row["metric"] in metrics
    )
    lines.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "Whole tasks, with every replicate and condition retained, are resampled together. Replicates are not counted as independent tasks. Brier and ECE target the binary all-hidden-tests-pass event; false approval is conditional on a hidden-test failure. ECE differences are recomputed in each bootstrap draw using ten equal-width bins.",
            "",
            "This analysis does not measure how evaluator feedback changes candidate quality. The frozen corpus was formed from every source shared round-0 candidate without outcome-based filtering. Sparse suites, incomplete specifications, contamination, oracle errors, correlated model biases, unequal evaluator compute and a small task sample still limit inference. Intervals are exploratory and not multiplicity-adjusted.",
        ]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
