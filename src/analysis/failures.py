"""Privacy-preserving, explicitly heuristic high-gap case reports."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.metrics.core import Row, normalized_rows

KNOWN_STATUSES = {
    "passed",
    "failed",
    "timeout",
    "runtime_error",
    "memory_limit",
    "syntax_error",
    "blocked",
    "error",
    "resource_limit",
    "assertion_failure",
    "exception",
}


def _safe_counts(value: Any) -> Row:
    if isinstance(value, str):
        value = json.loads(value or "{}")
    if not isinstance(value, dict):
        return {"summary": "No structured execution summary available"}
    total, passed = value.get("total"), value.get("passed")
    result = {
        key: int(number)
        for key, number in (("total", total), ("passed", passed))
        if isinstance(number, (float, int)) and number >= 0
    }
    if "total" in result and "passed" in result:
        result["failed"] = max(0, result["total"] - result["passed"])
    statuses = value.get("status_counts", {})
    if isinstance(statuses, dict):
        safe_statuses: dict[str, int] = {}
        for key, count in statuses.items():
            if isinstance(count, int) and count >= 0:
                label = str(key) if key in KNOWN_STATUSES else "other"
                safe_statuses[label] = safe_statuses.get(label, 0) + count
        result["status_counts"] = safe_statuses
    return result


def infer_category(row: Row) -> str:
    """A search aid, not a verified causal explanation of an evaluator failure."""
    text = (
        str(row.get("judge_reasoning_summary", "")) + " " + str(row.get("critic_feedback", ""))
    ).lower()
    counts = _safe_counts(row.get("hidden_test_result", {})).get("status_counts", {})
    if counts.get("timeout") or any(
        word in text for word in ("complexity", "quadratic", "exponential", "timeout")
    ):
        return "complexity failure"
    if any(word in text for word in ("off-by-one", "off by one", "index boundary")):
        return "off-by-one"
    if any(word in text for word in ("type error", "typeerror", "format", "return type")):
        return "type/format error"
    if any(word in text for word in ("edge case", "empty", "negative", "singleton", "boundary")):
        return "edge case missed"
    if any(word in text for word in ("assumption", "assumes", "precondition")):
        return "incorrect assumption"
    if any(
        word in text for word in ("explanation", "comment claims", "looks convincing", "narrative")
    ):
        return "evaluator trusted explanation over code"
    if any(word in text for word in ("algorithm", "recurrence", "logic error")):
        return "algorithmic error"
    return "evaluator failed to simulate behavior"


def _fence(value: str) -> str:
    longest = max((len(match) for match in re.findall(r"`+", value)), default=0)
    marker = "`" * max(4, longest + 1)
    return f"{marker}python\n{value}\n{marker}"


def generate_failure_cases(rows: list[Row], output_dir: Path) -> list[Row]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_index = output_dir / "index.json"
    previous = json.loads(previous_index.read_text()) if previous_index.exists() else []
    data = normalized_rows(rows)
    cases: list[Row] = []
    for row in sorted(
        data,
        key=lambda r: (
            -(r["judge_score"] - r["ground_truth_score"]),
            str(r["task_id"]),
            str(r["condition"]),
            r["round"],
        ),
    ):
        if row["judge_score"] < 0.8 or row["ground_truth_score"] > 0.5:
            continue
        case = {
            key: row.get(key)
            for key in (
                "experiment_id",
                "run_kind",
                "task_id",
                "condition",
                "replicate",
                "round",
                "problem",
                "candidate_code",
                "judge_score",
                "ground_truth_score",
                "judge_binary",
                "judge_reasoning_summary",
                "critic_feedback",
            )
        }
        case["evaluation_gap"] = row["judge_score"] - row["ground_truth_score"]
        case["hidden_failure_summary"] = _safe_counts(row.get("hidden_test_result", {}))
        case["likely_failure_category"] = infer_category(row)
        case["category_status"] = (
            "Unverified heuristic; human inspection required before causal claims"
        )
        identity = "|".join(
            str(row.get(key, ""))
            for key in ("experiment_id", "task_id", "condition", "replicate", "round")
        )
        filename = hashlib.sha256(identity.encode()).hexdigest()[:16]
        case["report_file"] = f"{filename}.md"
        mock = (
            "MOCK VALIDATION — synthetic case, not an empirical model failure.\n\n"
            if row.get("run_kind") == "mock"
            else ""
        )
        content = (
            f"# High-gap example: {row['task_id']}\n\n{mock}"
            f"Condition: {row['condition']}; round: {row['round']}. "
            f"J = {row['judge_score']:.3f}; hidden-test fraction G = {row['ground_truth_score']:.3f}; gap = {case['evaluation_gap']:.3f}.\n\n"
            f"## Problem\n\n{row.get('problem', '')}\n\n"
            f"## Candidate solution\n\n{_fence(str(row.get('candidate_code', '')))}\n\n"
            f"## Evaluator justification\n\n{row.get('judge_reasoning_summary', '')}\n\n"
            f"## Critic output\n\n{row.get('critic_feedback') or 'No critic in this condition.'}\n\n"
            f"## Hidden execution summary\n\n`{json.dumps(case['hidden_failure_summary'], sort_keys=True)}`\n\n"
            "Only aggregate status counts are shown. Test contents, actual/expected values, exception messages and tracebacks are withheld.\n\n"
            f"## Tentative category\n\n{case['likely_failure_category']}. {case['category_status']}.\n"
        )
        (output_dir / case["report_file"]).write_text(content, encoding="utf-8")
        cases.append(case)
    current_names = {case["report_file"] for case in cases}
    for old_case in previous:
        name = old_case.get("report_file", "")
        if re.fullmatch(r"[0-9a-f]{16}\.md", name) and name not in current_names:
            (output_dir / name).unlink(missing_ok=True)
    (output_dir / "index.json").write_text(
        json.dumps(cases, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# High-gap failure cases",
        "",
        "Threshold: J ≥ 0.8 and hidden-test fraction G ≤ 0.5. All rounds are eligible; repeated rounds are descriptive, not independent observations.",
        "",
    ]
    if any(r.get("run_kind") == "mock" for r in data):
        lines.extend(
            ["MOCK VALIDATION ONLY. These synthetic cases do not support research claims.", ""]
        )
    if not cases:
        lines.append(
            "No saved candidate met the prespecified threshold. This does not establish that evaluator blind spots are absent."
        )
    else:
        lines.extend(
            [
                "| Task | Condition | Round | J | G | Tentative category |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        lines.extend(
            f"| [{case['task_id']}]({case['report_file']}) | {case['condition']} | {case['round']} | {case['judge_score']:.3f} | {case['ground_truth_score']:.3f} | {case['likely_failure_category']} |"
            for case in cases
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cases
