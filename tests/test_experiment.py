from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.execution.sandbox import ExecutionResult, Executor, SandboxUnavailable
from src.experiments.runner import derive_replicate_seed, resolve_config, run_experiment
from src.models.base import CachedModel
from src.utils.serialization import read_completed_results, read_results, write_results

ROOT = Path(__file__).resolve().parents[1]


class SerializationTests(unittest.TestCase):
    def test_replicate_config_and_seed_streams(self):
        self.assertEqual(resolve_config({}, ROOT)["replicates"], 1)
        for invalid in (0, -1, 1.5, True, "2"):
            with self.assertRaises(ValueError):
                resolve_config({"replicates": invalid}, ROOT)
        self.assertEqual(derive_replicate_seed(42, 0), derive_replicate_seed(42, 0))
        self.assertNotEqual(derive_replicate_seed(42, 0), derive_replicate_seed(42, 1))

    def test_csv_roundtrip_preserves_types_and_newlines(self):
        rows = [
            {
                "experiment_id": "fixture",
                "task_id": "x",
                "round": 1,
                "seed": 42,
                "judge_score": 0.8,
                "ground_truth_score": 0.5,
                "judge_binary": True,
                "pass_at_1": False,
                "candidate_code": 'def f():\n    return "a,b"\n',
                "public_test_result": {"passed": 1},
                "hidden_test_result": {"passed": 0},
                "ensemble_scores": [0.8, 0.7],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            write_results(path, rows)
            self.assertEqual(read_results(path), rows)

    def test_bad_boolean_is_not_silently_truthy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("judge_binary\nfalse-ish\n")
            with self.assertRaises(ValueError):
                read_results(path)

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") and importlib.util.find_spec("pandas"),
        "Parquet extra not installed",
    )
    def test_parquet_full_results_can_be_analyzed(self):
        from src.analysis.pipeline import analyze

        rows = [
            {
                "experiment_id": "parquet",
                "run_kind": "mock",
                "task_id": task,
                "condition": condition,
                "round": 0,
                "judge_score": 0.8,
                "ground_truth_score": truth,
                "pass_at_1": truth == 1,
                "judge_binary": True,
                "ensemble_scores": [0.7, 0.9] if condition == "ensemble" else [],
                "ensemble_judgments": [
                    {"approve": True, "justification": "fixture", "probability_correct": 0.8}
                ]
                if condition == "ensemble"
                else [],
                "ensemble_mean": 0.8 if condition == "ensemble" else None,
                "ensemble_median": 0.8 if condition == "ensemble" else None,
                "ensemble_variance": 0.01 if condition == "ensemble" else None,
                "ensemble_majority_vote": True if condition == "ensemble" else None,
                "public_test_result": {"total": 1, "passed": 1, "status_counts": {"passed": 1}},
                "hidden_test_result": {
                    "total": 2,
                    "passed": int(truth * 2),
                    "status_counts": {"passed": 2} if truth == 1 else {"assertion_failure": 2},
                },
            }
            for task, truth in (("a", 1), ("b", 0))
            for condition in ("baseline", "ensemble")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.parquet"
            write_results(path, rows)
            restored = read_results(path)
            self.assertEqual(restored, rows)
            result = analyze(
                restored, Path(directory) / "analysis", bootstrap_samples=10, make_figures=False
            )
            self.assertEqual(len(result["summary"]), 2)

    def test_exclusions_preserve_seeded_order(self):
        from src.data.base import load_tasks

        original = load_tasks("toy", limit=10, seed=42)
        removed = original[1].task_id
        filtered = load_tasks("toy", limit=9, seed=42, exclude_task_ids=[removed])
        self.assertEqual(filtered, [task for task in original if task.task_id != removed])

    def test_partial_and_failed_manifests_cannot_be_analyzed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_results(root / "results.csv", [{"experiment_id": "x", "round": 0}])
            (root / "manifest.json").write_text(json.dumps({"status": "failed"}))
            with self.assertRaises(ValueError):
                read_completed_results(root / "results.csv")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "experiment_id": "x",
                        "expected_rows": 2,
                        "actual_rows": 1,
                    }
                )
            )
            with self.assertRaises(ValueError):
                read_completed_results(root / "results.csv")
            with self.assertRaises(ValueError):
                read_completed_results(root / "partial.csv")


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            Executor().check_available()
        except SandboxUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def config(self, cache: Path) -> dict:
        return {
            "seed": 42,
            "dataset": {"name": "toy", "limit": 2},
            "models": {"backend": "mock"},
            "rounds": 1,
            "conditions": ["baseline", "ensemble", "baseline_no_feedback"],
            "ensemble_size": 3,
            "cache_dir": str(cache),
            "execution": {"timeout_seconds": 1},
        }

    def test_mock_end_to_end_shared_initial_and_sanitized_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = run_experiment(
                self.config(root / "cache"), ROOT, output_dir=root / "run", update_latest=False
            )
            rows = read_results(path)
            self.assertEqual(len(rows), 12)
            for task_id in {row["task_id"] for row in rows}:
                initial = [r for r in rows if r["task_id"] == task_id and r["round"] == 0]
                self.assertEqual(len({r["candidate_code"] for r in initial}), 1)
            self.assertTrue(all(r["run_kind"] == "mock" for r in rows))
            for row in rows:
                self.assertEqual(
                    row["evaluation_gap"], row["judge_score"] - row["ground_truth_score"]
                )
                self.assertLessEqual(
                    set(row["hidden_test_result"]),
                    {
                        "total",
                        "passed",
                        "status_counts",
                        "duration_seconds",
                        "score",
                        "all_passed",
                        "oracle_valid",
                        "protocol",
                    },
                )
                self.assertTrue(row["ground_truth_valid"])
                self.assertEqual(row["experiment_type"], "iterative")
                self.assertEqual(len(row["public_task_sha256"]), 64)
            manifest = json.loads((path.parent / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["actual_rows"], manifest["expected_rows"])
            self.assertEqual(len(list((path.parent / "trajectories").glob("*.json"))), 6)
            with self.assertRaises(FileExistsError):
                run_experiment(
                    self.config(root / "cache"), ROOT, output_dir=root / "run", update_latest=False
                )

    def test_replicates_keep_shared_initials_and_separate_cache_draws_and_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root / "cache")
            cfg["replicates"] = 2
            path = run_experiment(cfg, ROOT, output_dir=root / "run", update_latest=False)
            rows = read_results(path)
            self.assertEqual(len(rows), 24)
            self.assertEqual({row["replicate"] for row in rows}, {0, 1})
            self.assertEqual(
                {row["replicate_seed"] for row in rows},
                {derive_replicate_seed(42, 0), derive_replicate_seed(42, 1)},
            )
            for task_id in {row["task_id"] for row in rows}:
                for replicate in (0, 1):
                    initial = [
                        r
                        for r in rows
                        if r["task_id"] == task_id
                        and r["replicate"] == replicate
                        and r["round"] == 0
                    ]
                    self.assertEqual(len(initial), 3)
                    self.assertEqual(len({r["candidate_code"] for r in initial}), 1)
            files = list((path.parent / "trajectories").glob("*.json"))
            self.assertEqual(len(files), 12)
            for replicate in (0, 1):
                self.assertEqual(sum(f"replicate-{replicate}-" in file.name for file in files), 6)
            calls = [
                json.loads(line) for line in (root / "cache/calls.jsonl").read_text().splitlines()
            ]
            initial_calls = [
                call
                for call in calls
                if call["role"] == "proposer" and call["call_id"].endswith("/initial")
            ]
            self.assertEqual(len(initial_calls), 4)
            self.assertEqual(len({call["request_hash"] for call in initial_calls}), 4)
            self.assertEqual(len({call["seed"] for call in initial_calls}), 4)
            self.assertTrue(all(not call["cache_hit"] for call in initial_calls))
            manifest = json.loads((path.parent / "manifest.json").read_text())
            self.assertEqual(manifest["replicates"], 2)
            self.assertEqual(manifest["actual_rows"], manifest["expected_rows"])

    def test_failed_run_keeps_partial_rows_and_failure_manifest(self):
        original = CachedModel.complete

        def fail_revision(client, messages, **kwargs):
            if "/revision/" in kwargs.get("call_id", ""):
                raise RuntimeError("Injected provider outage")
            return original(client, messages, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root / "cache")
            cfg["conditions"] = ["baseline"]
            with (
                patch.object(CachedModel, "complete", fail_revision),
                self.assertRaises(RuntimeError),
            ):
                run_experiment(cfg, ROOT, output_dir=root / "run", update_latest=False)
            manifest = json.loads((root / "run/manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len(read_results(root / "run/partial.csv")), 1)
            self.assertFalse((root / "run/results.csv").exists())

    def test_unsupported_oracle_is_saved_explicitly_and_analysis_rejects_it(self):
        from src.analysis.pipeline import analyze

        unsupported = ExecutionResult(1, 0, {"unsupported_protocol": 1}, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root / "cache")
            cfg.update(rounds=0, conditions=["baseline"])
            with patch.object(Executor, "run", return_value=unsupported):
                path = run_experiment(cfg, ROOT, output_dir=root / "run", update_latest=False)
            rows = read_results(path)
            self.assertTrue(all(not row["ground_truth_valid"] for row in rows))
            self.assertTrue(all(not row["pass_at_1"] for row in rows))
            with self.assertRaisesRegex(ValueError, "oracle"):
                analyze(rows, root / "analysis", bootstrap_samples=10, make_figures=False)


if __name__ == "__main__":
    unittest.main()
