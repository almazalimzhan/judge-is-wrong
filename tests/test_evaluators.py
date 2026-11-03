from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from src.evaluators.base import EvaluationParseError, Evaluator, parse_critic_json, parse_judge_json
from src.models.base import CachedModel
from src.models.mock import MockBackend
from src.proposers.base import Proposer


@dataclass
class PublicTask:
    task_id: str = "test/task"
    problem: str = "Return the sum of two integers as add(a, b)."
    entry_point: str = "add"
    hidden_tests: str = "PRIVATE_HIDDEN_TEST_SENTINEL"

    def public_view(self):
        return {
            "task_id": self.task_id,
            "problem": self.problem,
            "entry_point": self.entry_point,
            "public_tests": ["assert add(1, 2) == 3"],
        }


class RecordingModel:
    def __init__(self):
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        operation = json.loads(messages[-1]["content"])["operation"]
        if operation == "critique":
            return '{"summary": "Check negative inputs."}'
        if operation == "propose":
            return "def add(a, b):\n    return a + b\n"
        return '{"probability_correct": 0.8, "approve": true, "justification": "Looks plausible."}'


class EvaluatorTests(unittest.TestCase):
    def test_valid_json(self):
        value = parse_judge_json('{"probability_correct":0.8,"approve":true,"justification":"ok"}')
        self.assertEqual(value.probability_correct, 0.8)
        self.assertIs(value.approve, True)

    def test_invalid_judge_outputs_are_rejected(self):
        cases = [
            "```json\n{}\n```",
            "[]",
            "{}",
            '{"probability_correct":0.8,"probability_correct":0.9,"approve":true,"justification":"ok"}',
            '{"probability_correct":NaN,"approve":true,"justification":"ok"}',
            '{"probability_correct":true,"approve":true,"justification":"ok"}',
            '{"probability_correct":1.1,"approve":true,"justification":"ok"}',
            '{"probability_correct":0.8,"approve":"true","justification":"ok"}',
            '{"probability_correct":0.8,"approve":true,"justification":""}',
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(EvaluationParseError):
                parse_judge_json(text)

    def test_critic_requires_structured_summary(self):
        self.assertEqual(parse_critic_json('{"summary":"Check boundary."}'), "Check boundary.")
        with self.assertRaises(EvaluationParseError):
            parse_critic_json("Free-form commentary")

    def test_all_conditions_exclude_hidden_information(self):
        task = PublicTask()
        for condition in sorted(Evaluator.CONDITIONS):
            model = RecordingModel()
            result = Evaluator(model, ensemble_size=3).evaluate(
                task,
                "def add(a,b): return a+b",
                condition,
                {
                    "total": 1,
                    "passed": 1,
                    "status_counts": {"passed": 1},
                    "hidden_tests": "OTHER_HIDDEN_SENTINEL",
                },
                call_id=condition,
            )
            text = json.dumps(model.calls)
            self.assertNotIn("PRIVATE_HIDDEN_TEST_SENTINEL", text)
            self.assertNotIn("OTHER_HIDDEN_SENTINEL", text)
            self.assertEqual(result.judge_score, 0.8)
            if condition == "ensemble":
                calls = [kwargs["call_id"] for _, kwargs in model.calls]
                self.assertEqual(len(set(calls)), 3)
                self.assertEqual(result.ensemble_variance, 0)
                self.assertEqual(result.ensemble_median, 0.8)
                self.assertTrue(result.ensemble_majority_vote)
            if condition != "execution":
                self.assertNotIn("public_execution_evidence", text)

    def test_execution_evidence_excludes_nondeterministic_timing(self):
        model = RecordingModel()
        evaluator = Evaluator(model)
        for duration in (0.1, 0.2):
            evaluator.evaluate(
                PublicTask(),
                "def add(a,b):return a+b",
                "execution",
                {
                    "total": 1,
                    "passed": 1,
                    "status_counts": {"passed": 1},
                    "duration_seconds": duration,
                },
                call_id="same",
            )
        self.assertEqual(model.calls[0][0], model.calls[1][0])
        self.assertNotIn("duration_seconds", json.dumps(model.calls))
        with tempfile.TemporaryDirectory() as name:
            cached = CachedModel(MockBackend(), Path(name), {"judge": "mock-judge"}, seed=42)
            evaluator = Evaluator(cached)
            for duration in (0.12, 0.98):
                evaluator.evaluate(
                    PublicTask(),
                    "def add(a,b):return a+b",
                    "execution",
                    {
                        "total": 1,
                        "passed": 1,
                        "status_counts": {"passed": 1},
                        "duration_seconds": duration,
                    },
                    call_id="stable",
                )
            calls = [
                json.loads(line) for line in (Path(name) / "calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual([call["cache_hit"] for call in calls], [False, True])

    def test_proposer_whitelists_feedback(self):
        model = RecordingModel()
        code = Proposer(model).generate(
            PublicTask(),
            [
                {
                    "candidate_code": "def add(a,b):return a+b",
                    "round": 0,
                    "judge_score": 0.8,
                    "hidden_test_result": "SENSITIVE_RESULT",
                    "ground_truth_score": "SENSITIVE_SCORE",
                }
            ],
            "test",
        )
        self.assertIn("def add", code)
        prompt = json.dumps(model.calls)
        self.assertNotIn("SENSITIVE", prompt)
        self.assertIn("candidate_code", prompt)

    def test_no_feedback_revision_does_not_require_scores(self):
        model = RecordingModel()
        Proposer(model).generate(PublicTask(), [{"candidate_code": "pass", "round": 0}], "test")
        sent = json.loads(model.calls[0][0][-1]["content"])
        self.assertEqual(sent["history"], [{"candidate_code": "pass", "round": 0}])


if __name__ == "__main__":
    unittest.main()
