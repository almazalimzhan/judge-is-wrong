from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.failures import generate_failure_cases
from src.analysis.pipeline import analyze, validate_trajectories
from src.analysis.report import render_report
from src.analysis.statistics import (
    paired_comparisons,
    paired_endpoint_comparisons,
    summarize_condition,
    task_cluster_samples,
)


def fixtures() -> list[dict]:
    return [
        {
            "experiment_id": "unit",
            "run_kind": "mock",
            "task_id": task,
            "condition": condition,
            "replicate": 0,
            "round": round_number,
            "judge_score": min(
                0.99, 0.5 + 0.2 * round_number + (0.1 if condition == "adversarial" else 0)
            ),
            "judge_binary": round_number == 1,
            "ground_truth_score": truth,
            "pass_at_1": truth == 1,
            "problem": "Write f(x).",
            "candidate_code": "def f(x): return x",
            "judge_reasoning_summary": "Possibly misses an edge case",
            "critic_feedback": "Check empty input",
            "hidden_test_result": {
                "total": 2,
                "passed": int(truth * 2),
                "status_counts": {"failed": int(2 - truth * 2)},
                "tests": "PRIVATE_CANARY",
                "stderr": "PRIVATE_CANARY",
            },
        }
        for task, truth in (("a", 0.5), ("b", 1), ("c", 0), ("d", 1))
        for condition in ("baseline", "adversarial")
        for round_number in (0, 1)
    ]


class AnalysisTests(unittest.TestCase):
    def test_cluster_resampling_keeps_conditions_and_rounds_together(self) -> None:
        original = fixtures()
        sample = next(task_cluster_samples(original, 1, 3))
        for task_id in {r["task_id"] for r in sample}:
            cluster = [r for r in sample if r["task_id"] == task_id]
            self.assertEqual(len(cluster), 4)
            self.assertEqual({r["round"] for r in cluster}, {0, 1})
            self.assertEqual({r["condition"] for r in cluster}, {"baseline", "adversarial"})

    def test_bootstrap_determinism_and_single_task_interval(self) -> None:
        rows = [r for r in fixtures() if r["condition"] == "baseline" and r["round"] == 1]
        self.assertEqual(summarize_condition(rows, 40, 9), summarize_condition(rows, 40, 9))
        result = summarize_condition(rows[:1], 40, 9)
        self.assertIsNotNone(result["mean_gap"])
        self.assertIsNone(result["mean_gap_ci_low"])

    def test_replicates_remain_in_task_clusters_and_do_not_create_precision(self) -> None:
        original = fixtures()
        repeated = [dict(row, replicate=replicate) for row in original for replicate in (0, 1)]
        sample = next(task_cluster_samples(repeated, 1, 3))
        for task_id in {row["task_id"] for row in sample}:
            cluster = [row for row in sample if row["task_id"] == task_id]
            self.assertEqual(len(cluster), 8)
            self.assertEqual({row["replicate"] for row in cluster}, {0, 1})
        baseline = [row for row in original if row["condition"] == "baseline" and row["round"] == 1]
        replicated = [
            row for row in repeated if row["condition"] == "baseline" and row["round"] == 1
        ]
        one = summarize_condition(baseline, samples=80, seed=21)
        two = summarize_condition(replicated, samples=80, seed=21)
        self.assertEqual(one["n_tasks"], two["n_tasks"])
        self.assertEqual(one["mean_gap"], two["mean_gap"])
        self.assertEqual(one["mean_gap_ci_low"], two["mean_gap_ci_low"])
        self.assertEqual(one["mean_gap_ci_high"], two["mean_gap_ci_high"])

    def test_replicate_grid_and_experiment_types_cannot_be_mixed(self) -> None:
        original = fixtures()
        repeated = original + [dict(row, replicate=1) for row in original]
        self.assertEqual(len(validate_trajectories(repeated)), 32)
        # A whole task-replicate is missing in every condition, so simple pairing
        # alone would miss this unequal weighting across tasks.
        missing = [row for row in repeated if not (row["task_id"] == "a" and row["replicate"] == 1)]
        with self.assertRaisesRegex(ValueError, "replicate set"):
            validate_trajectories(missing)
        for invalid in (
            [dict(row, experiment_type="frozen_candidates") for row in original],
            [
                dict(row, experiment_type="frozen_candidates") if i == 0 else row
                for i, row in enumerate(original)
            ],
            [
                dict(row, ground_truth_protocol="mbpp")
                if i == 0
                else dict(row, ground_truth_protocol="humaneval-plus")
                for i, row in enumerate(original)
            ],
        ):
            with self.assertRaises(ValueError):
                validate_trajectories(invalid)

    def test_paired_quality_metrics_include_binary_prediction_error_and_calibration(self) -> None:
        rows = [row for row in fixtures() if row["round"] == 1]
        metrics = {row["metric"] for row in paired_comparisons(rows, samples=20, seed=9)}
        self.assertTrue(
            {
                "brier_score",
                "false_approval_rate",
                "ece",
                "mean_binary_gap",
                "binary_prediction_error",
            }
            <= metrics
        )

    def test_paired_gap_effect_preserves_exact_difference(self) -> None:
        rows = [r for r in fixtures() if r["round"] == 1]
        comparisons = paired_comparisons(rows, samples=30, seed=9)
        gap = next(r for r in comparisons if r["metric"] == "mean_gap")
        self.assertAlmostEqual(gap["estimate"], 0.1)
        self.assertAlmostEqual(gap["ci_low"], 0.1)
        self.assertAlmostEqual(gap["ci_high"], 0.1)
        # Constant paired differences carry no estimable standardized effect.
        self.assertIsNone(gap["paired_standardized_effect_dz"])

    def test_duplicate_missing_and_mixed_runs_fail(self) -> None:
        rows = fixtures()
        for invalid in (
            rows + [rows[0]],
            rows[:-1],
            [{**r, "run_kind": "real"} if i == 0 else r for i, r in enumerate(rows)],
        ):
            with self.assertRaises(ValueError):
                validate_trajectories(invalid)

    def test_paired_differences_of_changes_include_no_feedback_control(self) -> None:
        rows = fixtures()
        control = [
            {**r, "condition": "baseline_no_feedback", "judge_score": 0.5}
            for r in rows
            if r["condition"] == "baseline"
        ]
        comparison = next(
            r
            for r in paired_endpoint_comparisons(rows + control, samples=20, seed=9)
            if r["condition"] == "baseline"
            and r["reference"] == "baseline_no_feedback"
            and r["metric"] == "delta_gap"
        )
        self.assertAlmostEqual(comparison["estimate"], 0.2)
        self.assertAlmostEqual(comparison["ci_low"], 0.2)
        self.assertAlmostEqual(comparison["ci_high"], 0.2)

    def test_analysis_final_round_serialization_and_safe_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = analyze(fixtures(), root, bootstrap_samples=30, seed=17, make_figures=False)
            self.assertEqual({r["round"] for r in result["summary"]}, {1})
            self.assertTrue((root / "paired_comparisons.csv").is_file())
            serialized = json.loads((root / "analysis.json").read_text())
            self.assertEqual(serialized["provenance"]["run_kind"], "mock")
            self.assertEqual(serialized["provenance"]["oracle_verification"], "legacy_unverified")
            report = render_report(fixtures(), root, root / "report.md")
            self.assertIn("Empirical Results: PENDING", report.read_text())
            self.assertIn("re-execute", report.read_text())
            failure_payload = "".join(
                path.read_text() for path in (root / "failure_cases").glob("*")
            )
            self.assertNotIn("PRIVATE_CANARY", failure_payload)

    def test_failure_categories_are_tentative_and_never_dump_hidden_payload(self) -> None:
        rows = fixtures()
        rows[0]["judge_score"] = 0.95
        with tempfile.TemporaryDirectory() as directory:
            cases = generate_failure_cases(rows, Path(directory))
            self.assertTrue(cases)
            self.assertEqual(cases[0]["likely_failure_category"], "edge case missed")
            self.assertIn("Unverified", cases[0]["category_status"])
            self.assertNotIn("PRIVATE_CANARY", json.dumps(cases))


if __name__ == "__main__":
    unittest.main()
