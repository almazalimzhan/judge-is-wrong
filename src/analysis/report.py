"""Compose a small paper-style report from saved, inspectable analysis artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.analysis.failures import generate_failure_cases
from src.analysis.pipeline import input_digest, validate_trajectories
from src.metrics.core import Row


def _formatted(row: Row, metric: str) -> str:
    value = row.get(metric)
    if value is None:
        return "undefined"
    low, high = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
    return (
        f"{value:.3f} [{low:.3f}, {high:.3f}]"
        if low is not None
        else f"{value:.3f} [CI unavailable]"
    )


def render_report(rows: list[Row], analysis_dir: Path, output_path: Path) -> Path:
    data = validate_trajectories(rows)
    analysis_dir, output_path = Path(analysis_dir), Path(output_path)
    analysis = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    provenance = analysis["provenance"]
    if provenance["experiment_id"] != data[0].get("experiment_id") or provenance[
        "input_sha256"
    ] != input_digest(data):
        raise ValueError(
            "Analysis provenance does not match the requested experiment; rerun analysis"
        )
    cases = generate_failure_cases(data, analysis_dir / "failure_cases")
    mock = provenance["run_kind"] == "mock"
    status = (
        "**Empirical Results: PENDING.** The tables below are synthetic mock-pipeline validation only. No real proposer or judge was measured."
        if mock
        else "These measurements describe the configured models, tasks, sampling seed and feedback policy in this run. They do not establish intentional deception or general oversight failure."
    )
    oracle_notice = (
        "**Ground-truth provenance: legacy/unverified.** " + provenance["oracle_warning"]
        if provenance.get("oracle_verification") == "legacy_unverified"
        else "Ground-truth execution protocol: `function-rpc-v1` with explicitly valid execution summaries. Protocol verification does not establish exhaustive semantic correctness."
    )
    abstract = (
        "We provide a reproducible code-generation experiment to test whether optimizing an LLM evaluator's feedback increases evaluator score faster than hidden-test correctness. "
        + (
            "The current run validates the pipeline with deterministic mock models; empirical hypotheses remain untested."
            if mock
            else "The descriptive estimates below compare the configured feedback policies using held-out test execution and task-cluster uncertainty. Interpret them within the recorded sample and configuration."
        )
    )
    descriptions = {
        "baseline": "Single judge feedback",
        "baseline_no_feedback": "Single-judge measurement with iterative proposer revision without evaluator feedback (control)",
        "critique": "Ordinary critic followed by a judge",
        "adversarial": "Adversarial critic followed by a judge",
        "ensemble": "Ensemble of separately sampled judge calls",
        "execution": "Judge with public-test execution evidence",
        "execution_evidence": "Judge with public-test execution evidence",
    }
    condition_list = "\n".join(
        f"- `{condition}`: {descriptions.get(condition, 'Configured evaluator policy')}"
        for condition in provenance["conditions"]
    )

    def figure(name: str, caption: str) -> str:
        filename = name + ".png"
        if not provenance.get("figures_generated") or filename not in analysis.get(
            "figure_files", []
        ):
            return ""
        target = analysis_dir / "figures" / filename
        if not target.is_file():
            return ""
        relative = Path(os.path.relpath(target, output_path.parent)).as_posix()
        return f"![{caption}]({relative})\n\n"

    result_figures = "".join(
        figure(name, caption)
        for name, caption in (
            ("judge_vs_ground_truth", "Judge score versus hidden-test fraction, final round"),
            (
                "mean_gap_by_condition",
                "Mean final-round evaluation gap and 95% task-bootstrap intervals",
            ),
            ("scores_across_rounds", "Mean judge score and hidden-test fraction across revisions"),
            ("gap_across_rounds", "Evaluation gap across proposer revisions"),
            ("reliability_curve", "Reliability for the all-hidden-tests-pass event"),
            (
                "false_approval_by_condition",
                "Conditional false approval rate by evaluator condition",
            ),
            ("gap_distribution", "Distribution of final-round evaluation gaps"),
        )
    )
    failure_figure = figure(
        "task_trajectory_examples",
        "Selected per-task trajectories; diagnostic examples are not representative",
    )
    table = [
        "| Condition | Mean J | Mean G | Mean J − G | False approval | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["summary"]:
        table.append(
            "| "
            + str(row["condition"])
            + " | "
            + " | ".join(
                _formatted(row, key)
                for key in (
                    "mean_judge_score",
                    "mean_ground_truth",
                    "mean_gap",
                    "false_approval_rate",
                    "brier_score",
                    "ece",
                )
            )
            + " |"
        )
    content = f"""# When the Judge Is Wrong

## Abstract

{abstract}

## 1. Introduction

Can a proposer optimize a fallible evaluator while objective task quality fails to improve? This repository studies that proxy-optimization question on small Python programming tasks. It is a controlled proxy experiment, not a solution to scalable oversight.

## 2. Related Motivation

Weak supervision uses imperfect signals in place of direct ground truth. Reward hacking and evaluator exploitation ask whether optimizing such signals exposes their systematic errors. Adversarial evaluation, independent assessments and execution evidence are candidate mitigations. This project operationalizes familiar questions rather than claiming a novel oversight paradigm; source references and distinctions appear in the repository README.

## 3. Experimental Setup

For each task, shared initial code x₀ branches into condition-specific revision trajectories through round T. Feedback-enabled proposers receive only public task information and evaluator feedback. A configured no-feedback control revises without evaluator feedback. Hidden-test execution supplies G, the fraction of hidden test units passed, and a binary all-tests-pass event. J is the evaluator's predicted probability of that binary event. The diagnostic evaluation gap is J − G. Because these targets differ, that gap is not itself a calibration error.

Experiment: `{provenance["experiment_id"]}`. Run kind: `{provenance["run_kind"]}`. Tasks: {provenance["n_tasks"]}. Proposer replicates per task: {provenance["n_replicates_per_task"]}. Final round: {provenance["final_round"]}. Models, backend, dataset digest, seeds and prompts must be read from the corresponding run manifest and call cache.

{oracle_notice}

## 4. Evaluator Conditions

Conditions actually included in this run:

{condition_list}

Same-model ensemble calls can remain correlated. Condition-specific revisions estimate the combined feedback-and-optimization policy, not the isolated effect of replacing a judge on fixed candidates.

## 5. Metrics

Primary summaries use final-round candidates only. Conditional false approval is P(approve | at least one hidden test unit fails); false rejection is P(reject | all units pass). Brier score and ten-bin ECE compare J to the binary all-tests-pass event. Trajectory metrics compare round 0 with T. ROC-AUC and average precision are omitted when only one ground-truth class is present. Undefined quantities remain missing.

Intervals are 95% percentile task-cluster bootstrap intervals ({provenance["bootstrap_samples"]} draws, seed {provenance["seed"]}); all proposer replicates, rounds and conditions for a sampled task remain together. The analysis requires the same replicate set for every task; point estimates weight tasks equally through this balanced grid. Additional replicates do not create additional independent task clusters. Final-score paired differences and task-standardized gap effects are recorded in `paired_comparisons.csv`. Paired differences of round-0-to-final changes appear in `endpoint_comparisons.csv`, including comparisons to `baseline_no_feedback` when configured. Within-ensemble score variance measures member disagreement, not the estimator's sampling variance. No multiplicity-adjusted significance claim is made. Intervals are unavailable with fewer than two tasks or too many undefined bootstrap draws.

## 6. Results

{status}

Values show point estimate [95% task-bootstrap CI]. A missing interval is not zero uncertainty.

{chr(10).join(table)}

The final-round table does not test H1 by itself. Consult `trajectories.csv` for the paired change in J, G and J − G from round 0 to T, including tasks where J increases while G falls or stays fixed. Contrasts against baseline and their intervals are descriptive and exploratory.

{result_figures}

## 7. Failure Analysis

The prespecified screen J ≥ 0.8 and G ≤ 0.5 identified {len(cases)} saved round-level cases. Repeated rounds of one task are not independent failures. Aggregate hidden-test status counts are available in `failure_cases/`; hidden inputs, expected values, source and exception text are withheld. Failure categories are keyword-based hypotheses requiring human verification. Cases are selected for diagnosis and cannot estimate population prevalence.

{failure_figure}

## 8. Limitations

Code-generation tasks are only a proxy for open-ended oversight. Tests are incomplete specifications; deterministic public/hidden partitions are small and can share structure. Synthetic evaluator imperfections are much simpler than real oversight failures. Model families may share biases, repeated calls need not be independent, and benchmark contamination may affect results. LLM-as-judge is not equivalent to human oversight. Small samples limit generalization. Execution isolation limits depend on the recorded backend; the harness is not designed as a security boundary against a malicious host or adversarial Python interpreter. Mock results test software behavior only.

## 9. Discussion

An increased gap would be evidence of evaluation mismatch within this task distribution and feedback policy. It would not establish that advanced models deceive humans. A reduced gap under an intervention would not by itself identify the mechanism: both revised candidates and judge decisions can change. The separate frozen-candidate experiment compares evaluator quality on identical code. This iterative report instead describes feedback-driven trajectories; its task-cluster intervals retain all seeded proposer replicates. The no-feedback control helps compare revision policies, and cross-family evaluators can probe shared errors.

## 10. Future Work

Run the prespecified 50–100-task study with real models; repeat with independent seeds; compare weaker and stronger judges, same-family and cross-family evaluators, and prompt perturbations. Use the separate frozen-candidate experiment to separate judge effects from candidate changes; retain the implemented no-feedback control and seeded proposer replicates. Audit selected failures manually without releasing held-out tests.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
