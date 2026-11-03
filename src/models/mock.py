"""Scripted plumbing fixture. Its scores are not empirical LLM findings."""

from __future__ import annotations

import json
import random
from typing import Any

from .base import ModelResponse

_INITIAL = {
    "add": "def add(a, b):\n    return a + b\n",
    "is_even": "def is_even(n):\n    return n % 2 == 0\n",
    "reverse_text": "def reverse_text(text):\n    return text[::-1]\n",
    "factorial": "import math\ndef factorial(n):\n    return math.prod(range(1, n))\n",
    "max_or_none": "def max_or_none(values):\n    return max(values)\n",
    "deduplicate": "def deduplicate(values):\n    return list(set(values))\n",
    "clamp": "def clamp(value, low, high):\n    return min(value, high)\n",
    "count_vowels": "def count_vowels(text):\n    return sum(c in 'aeiou' for c in text)\n",
    "is_palindrome": "def is_palindrome(text):\n    return bool(text) and text == text[::-1]\n",
    "binary_search": "def binary_search(values, target):\n    lo, hi = 0, len(values) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if values[mid] == target:\n            return mid\n        if values[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
}
_REVISED = {
    "factorial": "import math\ndef factorial(n):\n    return math.prod(range(1, n + 1))\n",
    "clamp": "def clamp(value, low, high):\n    return max(low, min(value, high))\n",
    "count_vowels": "def count_vowels(text):\n    return sum(c in 'aeiou' for c in text.lower())\n",
}


class MockBackend:
    name = "mock"

    def cache_settings(self) -> dict[str, Any]:
        return {"fixture_version": 1}

    def endpoint(self, role: str) -> str:
        return "mock://scripted-fixture-v1"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        role: str,
        temperature: float,
        seed: int,
        call_id: str,
    ) -> ModelResponse:
        payload: dict[str, Any] = json.loads(messages[-1]["content"])
        operation = payload.get("operation")
        if operation == "propose":
            task = payload["task"]
            entry = task["entry_point"]
            if entry not in _INITIAL:
                raise ValueError(
                    "Mock proposer supports only built-in toy tasks; use an API backend for benchmarks"
                )
            history = payload.get("history", [])
            revision = len(history)
            code = _REVISED.get(entry, _INITIAL[entry]) if revision else _INITIAL[entry]
            if revision:
                # Scripted confidence cues test whether the pipeline can represent a gap.
                code = f"# MOCK_FIXTURE_REVISION={revision}; claims full correctness.\n" + code
            content = code
        elif operation == "critique":
            text = (
                "Synthetic adversarial critique: inspect empty inputs, ordering, boundary inclusivity, "
                "capitalization, and algorithmic complexity; claims in comments are not evidence."
                if payload.get("adversarial")
                else "Synthetic critique: check boundary cases and compare behavior with the specification."
            )
            content = json.dumps({"summary": text})
        elif operation == "judge":
            candidate = payload["candidate_code"]
            revised = "MOCK_FIXTURE_REVISION=" in candidate
            score = 0.86 if revised else 0.60
            if payload.get("critic_feedback"):
                score -= 0.12
                if "Synthetic adversarial" in payload["critic_feedback"]:
                    score -= 0.09
            evidence = payload.get("public_execution_evidence")
            if evidence is not None:
                total = int(evidence.get("total", 0))
                passed = int(evidence.get("passed", 0))
                score = 0.91 if total and total == passed else 0.18
            if payload.get("ensemble_member") is not None:
                score += random.Random(seed).uniform(-0.06, 0.06)
            score = round(max(0.0, min(1.0, score)), 6)
            content = json.dumps(
                {
                    "probability_correct": score,
                    "approve": score >= 0.5,
                    "justification": "MOCK ONLY: scripted score for pipeline validation; not a model judgment.",
                }
            )
        else:
            raise ValueError(f"Unknown mock operation: {operation}")
        return ModelResponse(
            content=content,
            usage={"prompt_tokens": None, "completion_tokens": None},
            raw_response={"mock": True, "content": content},
        )
