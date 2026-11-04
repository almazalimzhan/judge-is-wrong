import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.data.base import load_tasks
from src.data.humaneval import _public_examples
from src.utils.serialization import read_completed_results, result_experiment_type, write_results


class StrongerProtocolTests(unittest.TestCase):
    def test_plus_retains_complete_expanded_harness_and_hides_it(self):
        record = {
            "task_id": "HumanEval/0",
            "entry_point": "inc",
            "prompt": 'def inc(x):\n    """Increment.\n    >>> inc(1)\n    2\n    """\n',
            "canonical_solution": "    return x + 1\n",
            "test": "def check(candidate):\n    for x in range(100):\n        assert candidate(x) == x + 1\n# HIDDEN_PLUS_SENTINEL",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plus.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps(record) + "\n")
            task = load_tasks("humaneval_plus", path, limit=1)[0]
            self.assertEqual(len(task.hidden_tests), 1)
            self.assertIn("range(100)", task.hidden_tests[0].code)
            self.assertIn("G is binary", task.metadata["ground_truth_unit"])
            self.assertIn("Apache-2.0", task.metadata["license"])
            self.assertNotIn("HIDDEN_PLUS_SENTINEL", json.dumps(task.public_view()))
            self.assertNotIn("return x + 1", json.dumps(task.public_view()))

    def test_plus_requires_a_file(self):
        with self.assertRaises(ValueError):
            load_tasks("humaneval_plus")

    def test_malformed_upstream_doctest_has_no_execution_evidence(self):
        prompt = 'def f(x):\n    """Identity.\n    >>> f("abc\\ndef")\n    "abc"\n    """\n'
        self.assertEqual(_public_examples(prompt), ())

    def test_frozen_latest_requires_its_own_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "latest_frozen.csv"
            write_results(
                path,
                [{"experiment_id": "frozen", "ground_truth_valid": False, "replicate_seed": 123}],
            )
            (root / "latest_frozen.manifest.json").write_text(json.dumps({"status": "failed"}))
            with self.assertRaises(ValueError):
                read_completed_results(path)
            (root / "latest_frozen.manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "experiment_id": "frozen",
                        "expected_rows": 1,
                        "actual_rows": 1,
                    }
                )
            )
            restored = read_completed_results(path)
            self.assertIs(restored[0]["ground_truth_valid"], False)
            self.assertEqual(restored[0]["replicate_seed"], 123)

    def test_experiment_types_cannot_be_pooled(self):
        self.assertEqual(result_experiment_type([{}]), "iterative")
        with self.assertRaises(ValueError):
            result_experiment_type([{}, {"experiment_type": "frozen_candidates"}])


if __name__ == "__main__":
    unittest.main()
