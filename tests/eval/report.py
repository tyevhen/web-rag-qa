import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.eval.dataset import content_hash
from tests.eval.runner import RunResult
from tests.eval.thresholds import RESULTS_DIR


def _per_question(r: RunResult) -> dict[str, Any]:
    refs = set(r.row.reference_context_hashes)
    return {
        "question": r.row.question,
        "category": r.row.category,
        "route": r.route,
        "retry_count": r.retry_count,
        "latency_s": r.latency_s,
        "sub_questions": r.sub_questions,
        "answer": r.answer,
        "reference": r.row.reference,
        "retrieved_ref_hits": sum(1 for c in r.retrieved if content_hash(c.content) in refs),
        "graded_ref_hits": sum(1 for c in r.graded if content_hash(c.content) in refs),
        "error": r.error,
    }


def _flatten(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not isinstance(node, dict):
        return [(prefix, node)]
    return [item for k, v in node.items() for item in _flatten(v, f"{prefix}.{k}" if prefix else k)]


def _markdown(report: dict[str, Any], violations: list[str]) -> str:
    lines = [f"# Eval report {report['meta']['timestamp']}", "", f"Judge: `{report['meta']['judge']}`  ",
             f"Rows: {report['meta']['rows']}  ", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for section in ("deterministic", "llm") for k, v in _flatten(report.get(section, {}), section)]
    lines += ["", "## Threshold violations", ""] + ([f"- {v}" for v in violations] or ["None"])
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], results: list[RunResult], violations: list[str]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / report["meta"]["timestamp"].replace(":", "-")
    full = {**report, "violations": violations, "questions": [_per_question(r) for r in results]}
    stem.with_suffix(".json").write_text(json.dumps(full, indent=2, ensure_ascii=False))
    md = _markdown(report, violations)
    stem.with_suffix(".md").write_text(md)
    print(md)
    return stem.with_suffix(".json")


def meta(rows: int, judge: str | None) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": rows,
        "judge": judge or "skipped",
    }

