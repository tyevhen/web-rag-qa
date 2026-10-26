"""Offline evaluation over the golden set.

    uv run python -m tests.eval.run_eval              # deterministic + Ragas LLM metrics
    uv run python -m tests.eval.run_eval --skip-llm   # deterministic only — free, CI-safe
    uv run python -m tests.eval.run_eval --limit 5    # smoke run

Exits 1 when any metric violates tests/eval/thresholds.json.
"""
import argparse
import asyncio
import logging
import sys
from typing import Any

from agent.config import get_agent_config
from agent.graph import build_graph
from db.client import create_pool
from db.repository import DocumentRepository
from evaluation.judge import judge_model_name
from tests.eval import deterministic, ragas_metrics, report
from tests.eval.dataset import load_golden
from tests.eval.runner import RunResult, run_all
from tests.eval.thresholds import check_thresholds


def _deterministic(results: list[RunResult]) -> dict[str, Any]:
    return {
        "retrieval": deterministic.retrieval(results),
        "grader_threshold_sweep": deterministic.grader(results),
        "citations": deterministic.citations(results),
        "routing": deterministic.routing(results),
        "operational": deterministic.operational(results),
    }


async def main(skip_llm: bool, limit: int | None) -> int:
    rows = load_golden()[:limit]
    pool = await create_pool()
    try:
        repo = DocumentRepository(pool)
        results = await run_all(build_graph(get_agent_config(), repo), repo, rows)
    finally:
        await pool.close()

    full: dict[str, Any] = {
        "meta": report.meta(len(rows), None if skip_llm else judge_model_name()),
        "deterministic": _deterministic(results),
    }
    if not skip_llm:
        full["llm"] = await ragas_metrics.run(results)

    violations = check_thresholds(full)
    path = report.write(full, results, violations)
    print(f"Full results → {path}")
    return 1 if violations else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("tests.eval").setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.skip_llm, args.limit)))
