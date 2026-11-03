"""MBPP original JSONL and sanitized JSON adapters."""

from __future__ import annotations

import ast
import hashlib
import json
import warnings
from pathlib import Path

from .base import Task, TestCase


def _interface(code: str, tests: list[str]) -> tuple[str, str]:
    with warnings.catch_warnings():
        # Some original MBPP regex strings use legacy unescaped backslashes.
        # Preserve their semantics while avoiding noisy adapter-load warnings.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(code)
    functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise ValueError("MBPP record has no function definitions")
    called = {
        node.func.id
        for test in tests
        for node in ast.walk(ast.parse(test))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    target = next((fn.name for fn in functions if fn.name in called), functions[0].name)
    # Expose function names/arguments, never reference implementation bodies.
    signatures = []
    for fn in functions:
        signature = f"def {fn.name}({ast.unparse(fn.args)})"
        if fn.returns is not None:
            signature += f" -> {ast.unparse(fn.returns)}"
        signatures.append(signature + ": ...")
    return target, "\n".join(signatures)


def load_mbpp(path: Path, seed: int = 42) -> list[Task]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    tasks = []
    for record in records:
        # MBPP's documented evaluation split; avoid training/prompt/validation IDs.
        if not 11 <= int(record["task_id"]) <= 510:
            continue
        task_id = f"mbpp/{record['task_id']}"
        tests: list[str] = record["test_list"]
        if len(tests) < 2:
            raise ValueError(f"{task_id} needs at least two tests for the split")
        public_index = min(
            range(len(tests)),
            key=lambda index: hashlib.sha256(
                f"mbpp-public-v1|{seed}|{task_id}|{index}".encode()
            ).hexdigest(),
        )
        entry_point, signatures = _interface(record["code"], [tests[public_index]])
        setup = record.get("test_setup_code", "") or ""
        imports = record.get("test_imports", []) or []
        if isinstance(imports, str):
            imports = [imports]
        setup = "\n".join([setup, *imports]).strip()
        statement = record.get("text", record.get("prompt"))
        if not isinstance(statement, str):
            raise ValueError(f"{task_id} has no problem statement")
        cases = [TestCase(code=test, setup=setup) for test in tests]
        tasks.append(
            Task(
                task_id=task_id,
                problem=statement + "\n\nRequired Python interface:\n" + signatures,
                entry_point=entry_point,
                public_tests=(cases[public_index],),
                hidden_tests=tuple(
                    test for index, test in enumerate(cases) if index != public_index
                ),
                metadata={
                    "source": "Google Research MBPP",
                    "license": "CC-BY-4.0",
                    "benchmark_split": "official-test-11-510",
                    "split": "mbpp-public-v1",
                    "split_seed": seed,
                    "public_test_index": public_index,
                },
            )
        )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Duplicate MBPP task ids")
    return tasks
