from __future__ import annotations

import unittest

from src.metrics.core import (
    as_bool,
    compute_metrics,
    evaluation_gap,
    reliability_bins,
    trajectory_metrics,
)


def row(
    task: str,
    score: float,
    truth: float,
    approved: bool,
    round_number: int = 0,
    condition: str = "baseline",
) -> dict:
    return {
        "task_id": task,
        "condition": condition,
        "round": round_number,
        "judge_score": score,
        "ground_truth_score": truth,
        "pass_at_1": truth == 1,
        "judge_binary": approved,
    }


class MetricTests(unittest.TestCase):
    def test_invalid_oracles_cannot_become_correctness_labels(self) -> None:
        base = row("a", 0.9, 1, True)
        invalid = [
            dict(base, ground_truth_valid=False),
            dict(base, ground_truth_valid="False"),
            dict(base, hidden_test_result={"oracle_valid": False}),
            dict(base, hidden_test_result={"status_counts": {"integrity_violation": 1}}),
            dict(base, hidden_test_result={"status_counts": {"unsupported_protocol": 1}}),
            dict(base, hidden_test_result='{"oracle_valid": false}'),
        ]
        for observation in invalid:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                compute_metrics([observation])

    def test_classification_denominators(self) -> None:
        rows = [
            row("a", 0.9, 1, True),
            row("b", 0.6, 0.25, True),
            row("c", 0.4, 1, False),
            row("d", 0.1, 0.5, False),
        ]
        result = compute_metrics(rows)
        self.assertEqual(result["binary_prediction_error"], 0.5)
        for key in (
            "false_approval_rate",
            "false_rejection_rate",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "approved_error_fraction",
        ):
            self.assertEqual(result[key], 0.5)
        self.assertEqual(result["false_positive_fraction"], 0.25)
        self.assertAlmostEqual(result["brier_score"], 0.185)
        self.assertAlmostEqual(result["ece"], 0.35)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["pr_auc"], 5 / 6)

    def test_calibration_uses_binary_event_while_gap_uses_fraction(self) -> None:
        result = compute_metrics([row("a", 0.8, 0.5, True)])
        self.assertAlmostEqual(result["mean_gap"], 0.3)
        self.assertAlmostEqual(result["mean_binary_gap"], 0.8)
        self.assertAlmostEqual(result["brier_score"], 0.64)
        self.assertAlmostEqual(result["ece"], 0.8)
        self.assertAlmostEqual(evaluation_gap(0.8, 0.5), 0.3)

    def test_auc_ties_are_threshold_groups(self) -> None:
        rows = [
            row("a", 0.9, 1, True),
            row("b", 0.9, 0, True),
            row("c", 0.1, 1, False),
            row("d", 0.1, 0, False),
        ]
        result = compute_metrics(rows)
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["pr_auc"], 0.5)

    def test_undefined_is_not_zero(self) -> None:
        result = compute_metrics([row("a", 0.1, 0, False)])
        self.assertEqual(result["false_approval_rate"], 0)
        for key in ("false_rejection_rate", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            self.assertIsNone(result[key])
        self.assertIsNone(compute_metrics([])["mean_gap"])

    def test_reliability_score_one_and_empty_bins(self) -> None:
        bins = reliability_bins([row("a", 1, 1, True), row("b", 0, 0, False)])
        self.assertEqual(bins[-1]["count"], 1)
        self.assertEqual(bins[0]["count"], 1)
        self.assertIsNone(bins[1]["empirical_accuracy"])

    def test_csv_booleans_are_not_python_string_truthiness(self) -> None:
        self.assertFalse(as_bool("False"))
        self.assertTrue(as_bool("true"))
        with self.assertRaises(ValueError):
            as_bool("no")

    def test_invalid_probabilities_and_ground_truth(self) -> None:
        for score in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.assertRaises(ValueError):
                compute_metrics([row("a", score, 0, True)])
        bad = row("a", 0.5, 0.5, True)
        bad["pass_at_1"] = True
        with self.assertRaises(ValueError):
            compute_metrics([bad])

    def test_trajectory_changes_and_all_three_quadrants(self) -> None:
        rows = []
        for task, final_truth in (("a", 0.25), ("b", 0.5), ("c", 1)):
            rows.extend([row(task, 0.4, 0.5, False, 0), row(task, 0.8, final_truth, True, 1)])
        result = trajectory_metrics(rows)[0]
        self.assertAlmostEqual(result["mean_delta_judge"], 0.4)
        self.assertAlmostEqual(result["mean_delta_ground_truth"], 1 / 12)
        self.assertAlmostEqual(result["mean_delta_gap"], 0.4 - 1 / 12)
        for key in (
            "fraction_judge_up_truth_down",
            "fraction_judge_up_truth_unchanged",
            "fraction_both_improve",
        ):
            self.assertAlmostEqual(result[key], 1 / 3)


if __name__ == "__main__":
    unittest.main()
