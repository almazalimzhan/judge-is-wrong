from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.frozen import analyze_frozen, render_frozen_report, validate_frozen
from src.analysis.pipeline import analyze
from src.data.base import load_tasks
from src.execution.sandbox import Executor, SandboxUnavailable
from src.experiments.frozen import (
    FROZEN_CONDITIONS,
    freeze_candidates,
    load_corpus,
    public_task_sha256,
    run_frozen_evaluation,
)
from src.experiments.runner import derive_replicate_seed, run_experiment
from src.proposers.base import Proposer
from src.utils.serialization import read_results

ROOT = Path(__file__).resolve().parents[1]


def source_rows() -> list[dict]:
    rows = []
    for index, task in enumerate(load_tasks("toy", limit=2, seed=42)):
        for replicate in range(2):
            for condition in ("baseline", "adversarial"):
                rows.append(
                    {
                        "experiment_id": "source",
                        "experiment_type": "iterative",
                        "run_kind": "mock",
                        "task_id": task.task_id,
                        "replicate": replicate,
                        "replicate_seed": derive_replicate_seed(42, replicate),
                        "seed": 42,
                        "condition": condition,
                        "round": 0,
                        "problem": task.problem,
                        "candidate_code": f"# fixture {index}/{replicate}\ndef f(): return 0\n",
                        "public_task_sha256": public_task_sha256(task),
                        "judge_score": 0.9,
                        "judge_binary": True,
                        "ground_truth_score": float(index),
                        "pass_at_1": bool(index),
                        "ground_truth_valid": True,
                        "ground_truth_protocol": task.metadata.get("split", "unspecified"),
                        "hidden_test_result": {
                            "total": 1,
                            "passed": index,
                            "oracle_valid": True,
                            "status_counts": {"passed" if index else "assertion_failure": 1},
                        },
                        "judge_reasoning_summary": "PRIVATE_JUDGE_FEEDBACK_CANARY",
                        "critic_feedback": "PRIVATE_CRITIC_CANARY",
                    }
                )
    return rows


def frozen_rows() -> list[dict]:
    with tempfile.TemporaryDirectory() as directory:
        path = freeze_candidates(source_rows(), Path(directory) / "corpus.jsonl")
        records, metadata = load_corpus(path)
    rows = []
    for index, record in enumerate(records):
        truth = float(index // 2)
        for condition in FROZEN_CONDITIONS:
            score = 0.7 if condition == "baseline" else 0.5
            rows.append(
                {
                    **record,
                    "experiment_id": "frozen",
                    "experiment_type": "frozen_candidates",
                    "run_kind": "mock",
                    "condition": condition,
                    "round": 0,
                    "corpus_sha256": metadata["corpus_sha256"],
                    "judge_score": score,
                    "judge_binary": condition == "baseline",
                    "ground_truth_score": truth,
                    "pass_at_1": truth == 1,
                    "ground_truth_valid": True,
                    "ground_truth_protocol": "toy-v1",
                    "public_test_result": {"total": 1, "passed": 1, "status_counts": {"passed": 1}},
                    "hidden_test_result": {
                        "total": 1,
                        "passed": int(truth),
                        "oracle_valid": True,
                        "status_counts": {"passed" if truth else "assertion_failure": 1},
                    },
                }
            )
    return rows


class FrozenCorpusTests(unittest.TestCase):
    def test_freeze_exports_all_initial_candidates_without_outcomes_or_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = freeze_candidates(source_rows(), Path(directory) / "corpus.jsonl")
            records, metadata = load_corpus(path)
            self.assertEqual(len(records), 4)
            self.assertEqual(metadata["n_tasks"], 2)
            self.assertEqual(metadata["replicates"], [0, 1])
            content = path.read_text()
            for forbidden in (
                "judge_score",
                "judge_binary",
                "hidden_test",
                "ground_truth",
                "critic_feedback",
                "PRIVATE_",
            ):
                self.assertNotIn(forbidden, content)
            with self.assertRaises(FileExistsError):
                freeze_candidates(source_rows(), path)

    def test_freeze_rejects_revisions_and_conflicting_shared_initials(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                freeze_candidates(source_rows(), Path(directory) / "x.jsonl", 1)
            bad = source_rows()
            bad[0]["candidate_code"] = "different initial"
            with self.assertRaises(ValueError):
                freeze_candidates(bad, Path(directory) / "x.jsonl")

    def test_corpus_hash_tampering_and_unbalanced_population_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = freeze_candidates(source_rows(), Path(directory) / "corpus.jsonl")
            original = path.read_text()
            path.write_text(original.replace("return 0", "return 1", 1))
            with self.assertRaises(ValueError):
                load_corpus(path)
            records = [json.loads(line) for line in original.splitlines()][:-1]
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            meta = json.loads(path.with_suffix(".manifest.json").read_text())
            meta["corpus_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            meta["n_candidates"] = len(records)
            path.with_suffix(".manifest.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_corpus(path)


class FrozenAnalysisTests(unittest.TestCase):
    def test_separate_paired_metrics_and_report(self):
        rows = frozen_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = analyze_frozen(rows, root, bootstrap_samples=30)
            metrics = {row["metric"] for row in result["paired_comparisons"]}
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
            self.assertEqual(result["provenance"]["n_tasks"], 2)
            self.assertEqual(result["provenance"]["n_candidates"], 4)
            self.assertFalse((root / "trajectories.csv").exists())
            report = render_frozen_report(rows, root, root / "report.md")
            self.assertIn("PENDING", report.read_text())
            self.assertIn("no proposer calls or revisions", report.read_text())
            with self.assertRaises(ValueError):
                analyze(rows, root, bootstrap_samples=10, make_figures=False)

    def test_candidate_or_truth_mismatch_and_invalid_oracle_are_rejected(self):
        for field, value in (
            ("candidate_code", "def g():return 0"),
            ("ground_truth_score", 0.5),
            ("ground_truth_valid", False),
            ("round", 1),
            ("ground_truth_protocol", "different-suite"),
            ("experiment_type", "iterative"),
            ("public_task_sha256", None),
            ("replicate", 1.0),
            ("replicate", True),
            ("corpus_sha256", "invalid"),
        ):
            rows = frozen_rows()
            rows[0][field] = value
            with self.assertRaises(ValueError, msg=field):
                validate_frozen(rows)
        rows = frozen_rows()
        with self.assertRaises(ValueError):
            validate_frozen(rows + [rows[0]])

    def test_replicates_are_task_clusters_not_independent_tasks(self):
        rows = frozen_rows()
        single = [row for row in rows if row["replicate"] == 0]
        with tempfile.TemporaryDirectory() as directory:
            left = analyze_frozen(single, Path(directory) / "single", bootstrap_samples=50)
            right = analyze_frozen(rows, Path(directory) / "repeated", bootstrap_samples=50)
            for a, b in zip(left["summary"], right["summary"], strict=True):
                for metric in ("brier_score", "ece", "binary_prediction_error"):
                    self.assertAlmostEqual(a[metric], b[metric])
                    self.assertAlmostEqual(a[f"{metric}_ci_low"], b[f"{metric}_ci_low"])
                    self.assertAlmostEqual(a[f"{metric}_ci_high"], b[f"{metric}_ci_high"])


class FrozenEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            Executor().check_available()
        except SandboxUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_mock_frozen_runner_has_no_proposer_calls_and_preserves_code_and_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {
                "seed": 42,
                "replicates": 2,
                "dataset": {"name": "toy", "limit": 2},
                "models": {"backend": "mock"},
                "rounds": 0,
                "conditions": ["baseline", "adversarial"],
                "cache_dir": str(root / "source-cache"),
            }
            source = run_experiment(cfg, ROOT, output_dir=root / "source", update_latest=False)
            corpus = freeze_candidates(read_results(source), root / "corpus.jsonl")
            cfg["conditions"] = sorted(FROZEN_CONDITIONS)
            cfg["cache_dir"] = str(root / "frozen-cache")
            with patch.object(
                Proposer, "generate", side_effect=AssertionError("proposer forbidden")
            ):
                output = run_frozen_evaluation(
                    cfg, ROOT, corpus, output_dir=root / "frozen", update_latest=False
                )
            rows = read_results(output)
            self.assertEqual(len(rows), 16)
            validate_frozen(rows)
            calls = [
                json.loads(line)
                for line in (root / "frozen-cache/calls.jsonl").read_text().splitlines()
            ]
            self.assertNotIn("proposer", {call["role"] for call in calls})
            self.assertTrue(all("frozen/" in call["call_id"] for call in calls))
            manifest = json.loads((output.parent / "manifest.json").read_text())
            self.assertEqual(manifest["proposer_calls"], 0)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["actual_rows"], manifest["expected_rows"])
            bad = {**cfg, "rounds": 1}
            with self.assertRaises(ValueError):
                run_frozen_evaluation(
                    bad, ROOT, corpus, output_dir=root / "bad", update_latest=False
                )


if __name__ == "__main__":
    unittest.main()
