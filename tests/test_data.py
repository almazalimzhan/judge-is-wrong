import json
import tempfile
import unittest
from pathlib import Path

from src.data.base import Task, TestCase, load_tasks
from src.data.humaneval import load_humaneval
from src.data.mbpp import load_mbpp


class DatasetTests(unittest.TestCase):
    def test_toy_loading_is_seeded_and_has_hidden_tests(self):
        first = load_tasks("toy", limit=10, seed=7)
        second = load_tasks("toy", limit=10, seed=7)
        self.assertEqual([task.task_id for task in first], [task.task_id for task in second])
        self.assertEqual(len({task.task_id for task in first}), 10)
        self.assertTrue(all(task.public_tests and task.hidden_tests for task in first))

    def test_public_view_is_an_explicit_allowlist(self):
        task = Task(
            "sentinel",
            "problem",
            "f",
            (TestCase("assert f(1)==2"),),
            (TestCase("HIDDEN_SENTINEL"),),
            {"canonical_solution": "SECRET_REFERENCE"},
        )
        view = json.dumps(task.public_view())
        self.assertNotIn("HIDDEN_SENTINEL", view)
        self.assertNotIn("SECRET_REFERENCE", view)
        self.assertEqual(
            set(task.public_view()), {"task_id", "problem", "entry_point", "public_tests"}
        )

    def test_mbpp_constructs_disjoint_reproducible_split(self):
        record = {
            "task_id": 11,
            "text": "Increment an integer.",
            "code": "def increment(x):\n    return x + 1\n",
            "test_list": [
                "assert increment(1)==2",
                "assert increment(2)==3",
                "assert increment(3)==4",
            ],
            "test_setup_code": "import math",
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "mbpp.jsonl"
            excluded = {**record, "task_id": 601}
            path.write_text(json.dumps(record) + "\n" + json.dumps(excluded) + "\n")
            tasks = load_mbpp(path, seed=42)
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            again = load_mbpp(path, seed=42)[0]
        self.assertEqual(task.public_tests, again.public_tests)
        self.assertEqual(len(task.public_tests), 1)
        self.assertEqual(len(task.hidden_tests), 2)
        self.assertFalse(set(task.public_tests) & set(task.hidden_tests))
        self.assertNotIn("return x + 1", task.problem)
        self.assertIn("def increment(x): ...", task.problem)
        self.assertEqual(task.entry_point, "increment")
        self.assertEqual(task.public_tests[0].setup, "import math")

    def test_humaneval_keeps_dependent_harness_intact(self):
        record = {
            "task_id": "HumanEval/0",
            "entry_point": "inc",
            "prompt": 'def inc(x):\n    """Increment.\n    >>> inc(1)\n    2\n    """\n',
            "test": "def check(candidate):\n    for x in [1, 2]:\n        assert candidate(x) == x + 1",
            "canonical_solution": "    return x + 1",
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "he.jsonl"
            path.write_text(json.dumps(record) + "\n")
            task = load_humaneval(path)[0]
        self.assertEqual(len(task.public_tests), 1)
        self.assertEqual(len(task.hidden_tests), 1)
        self.assertIn("for x in [1, 2]", task.hidden_tests[0].code)
        self.assertNotIn("candidate(x)", json.dumps(task.public_view()))

    def test_invalid_dataset_and_limit_fail(self):
        with self.assertRaises(ValueError):
            load_tasks("unknown")
        with self.assertRaises(ValueError):
            load_tasks("toy", limit=11)
        with self.assertRaises(ValueError):
            load_tasks("mbpp")


if __name__ == "__main__":
    unittest.main()
