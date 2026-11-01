import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

from langgraph.graph.state import CompiledStateGraph

from agent.embed import embed_query, get_encoder
from agent.graph import route_of
from agent.nodes.grader import grading_queries, score_chunks
from agent.nodes.retriever import make_retriever
from agent.state import AgentState, Chunk
from db.repository import DocumentRepository
from tests.eval.dataset import GoldenRow, content_hash

log = logging.getLogger(__name__)

_ABLATION_K = 6  # matches the retriever's final top-k so Hit@k is comparable


@dataclass
class RunResult:
    row: GoldenRow
    route: str = ""
    answer: str = ""
    sub_questions: list[str] = field(default_factory=list)
    retrieved: list[Chunk] = field(default_factory=list)
    graded: list[Chunk] = field(default_factory=list)
    retrieved_ce_scores: list[float] = field(default_factory=list)
    retry_count: int = 0
    latency_s: float = 0.0
    # Hashes of top-k chunks per retrieval variant, for the hybrid/decomposition ablation
    ablation: dict[str, list[str]] = field(default_factory=dict)
    error: str = ""


def _initial_state(row: GoldenRow, sub_questions: list[str] | None = None) -> AgentState:
    return {
        "question": row.question,
        "history": row.history,
        "sub_questions": sub_questions or [],
        "retrieved_chunks": [],
        "graded_chunks": [],
        "answer": "",
        "citations": [],
        "clarifying_questions": [],
        "retry_count": 0,
    }


async def _ablation(repo: DocumentRepository, row: GoldenRow) -> dict[str, list[str]]:
    dense = await repo.similarity_search(embed_query(row.question), _ABLATION_K, get_encoder().name)
    sparse = await repo.fulltext_search(row.question, _ABLATION_K)
    # Real retriever node with no sub-questions → hybrid over the raw question only
    hybrid = await make_retriever(repo)(_initial_state(row))
    return {
        "dense_only": [content_hash(r["content"]) for r in dense],
        "sparse_only": [content_hash(r["content"]) for r in sparse],
        "hybrid_no_decompose": [content_hash(c.content) for c in hybrid["retrieved_chunks"]],
    }


async def _run_one(graph: CompiledStateGraph, repo: DocumentRepository, row: GoldenRow) -> RunResult:
    t0 = time.monotonic()
    state: dict[str, Any] = {}
    first_pass: list[Chunk] | None = None
    # Stream to also capture the first retrieval: after a retry the final one includes the augmenter's rewrites
    async for item in graph.astream(_initial_state(row), stream_mode=["updates", "values"]):
        mode, chunk = cast(tuple[str, dict[str, Any]], item)
        if mode == "values":
            state = chunk
        elif "retrieve" in chunk and first_pass is None:
            first_pass = chunk["retrieve"]["retrieved_chunks"]
    latency = round(time.monotonic() - t0, 2)
    retrieved: list[Chunk] = state["retrieved_chunks"]
    queries = grading_queries(row.question, state["sub_questions"])
    ce_scores = await asyncio.to_thread(score_chunks, queries, retrieved) if retrieved else []
    return RunResult(
        row=row,
        route=route_of(state),
        answer=state["answer"],
        sub_questions=state["sub_questions"],
        retrieved=retrieved,
        graded=state["graded_chunks"],
        retrieved_ce_scores=[float(s) for s in ce_scores],
        retry_count=state["retry_count"],
        latency_s=latency,
        # Same single pass as the other variants, so the decomposition comparison excludes the retry
        ablation={**await _ablation(repo, row), "hybrid_decomposed_first_pass": [content_hash(c.content) for c in first_pass or []]},
    )


async def run_all(graph: CompiledStateGraph, repo: DocumentRepository, rows: list[GoldenRow]) -> list[RunResult]:
    # Sequential on purpose: the generator LLM is rate-limited
    results: list[RunResult] = []
    for i, row in enumerate(rows, 1):
        log.info("[%d/%d] %s: %.70s", i, len(rows), row.category, row.question)
        try:
            results.append(await _run_one(graph, repo, row))
        except Exception as exc:
            log.exception("run failed for %r", row.question)
            results.append(RunResult(row=row, error=f"{type(exc).__name__}: {exc}"))
    return results
