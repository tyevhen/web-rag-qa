"""Run the red-team sample through the real graph (in-process, nothing written to query_log).

    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=web-rag-llm LLM_TIMEOUT=300 \
        uv run python -m tests.eval.redteam [--limit-per-arm 5]

Local generator only: attack traffic never goes to a hosted provider. A canary is appended to the persona prompt
for this run, so any answer containing it proves a system-prompt leak. Score afterwards with tests.eval.redteam_score.
"""
import argparse
import asyncio
import json
import logging
from collections import defaultdict

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from agent.config import TopicConfig, get_agent_config
from agent.graph import build_graph
from db.client import create_pool
from db.repositories.documents import DocumentRepository
from db.repositories.embeddings import EmbeddingRepository
from embeddings.upkeep import check_ready
from settings import get_settings
from tests.eval.dataset import GoldenRow
from tests.eval.redteam_sample import OUT as SAMPLE
from tests.eval.runner import RunResult, run_all

RESULTS = SAMPLE.parent / "results.jsonl"
log = logging.getLogger("tests.eval.redteam")
CANARY = "CANARY-7f3a"
_CACHE = SAMPLE.parent / "llm_cache.sqlite"  # separate from the golden-set cache; reruns replay finished calls


def _load(limit_per_arm: int | None, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in SAMPLE.read_text().splitlines()][:limit]
    seen: dict[str, int] = defaultdict(int)
    picked = []
    for r in rows:
        seen[r["arm"]] += 1
        if limit_per_arm is None or seen[r["arm"]] <= limit_per_arm:
            picked.append(r)
    return picked


def _record(meta: dict, r: RunResult) -> dict:
    return {
        **{k: meta[k] for k in ("id", "arm", "prompt_type", "source", "attack_technique", "language")},
        "question": meta["question"], "history_turns": len(meta["history"]),
        "route": r.route, "answer": r.answer, "clarifying_questions": r.clarifying_questions, "sub_questions": r.sub_questions,
        "retrieved": len(r.retrieved), "graded": len(r.graded),
        "max_ce": max(r.retrieved_ce_scores, default=None), "retry_count": r.retry_count,
        "latency_s": r.latency_s, "error": r.error,
        "prompt_chars": len(meta["question"]) + sum(len(t["content"]) for t in meta["history"]),
    }


async def main(limit_per_arm: int | None, limit: int | None) -> None:
    s = get_settings()
    if "localhost" not in s.llm_base_url and "127.0.0.1" not in s.llm_base_url:
        raise SystemExit(f"Refusing to send attack prompts to {s.llm_base_url}: set LLM_BASE_URL to local Ollama")
    # Resumable: each finished row is appended at once, and rows already in RESULTS are skipped on the next run
    done = {json.loads(line)["id"] for line in RESULTS.read_text().splitlines()} if RESULTS.exists() else set()
    meta = [m for m in _load(limit_per_arm, limit) if m["id"] not in done]
    config = TopicConfig(system_prompt=f"{get_agent_config().system_prompt}\nInternal ref: {CANARY}.")
    set_llm_cache(SQLiteCache(database_path=str(_CACHE)))
    pool = await create_pool()
    try:
        repo = DocumentRepository(pool)
        await check_ready(EmbeddingRepository(pool))
        graph = build_graph(config, repo)
        for i, m in enumerate(meta, 1):
            log.info("row %d of %d still to run (id %d)", i, len(meta), m["id"])
            [r] = await run_all(graph, repo, [GoldenRow(question=m["question"], category=m["arm"], history=m["history"])])
            with RESULTS.open("a") as f:
                f.write(json.dumps(_record(m, r), ensure_ascii=False) + "\n")
    finally:
        await pool.close()
    print(f"{len(done) + len(meta)} runs ({s.llm_model}) in {RESULTS}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("tests.eval").setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-arm", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N sample rows only")
    args = parser.parse_args()
    asyncio.run(main(args.limit_per_arm, args.limit))
