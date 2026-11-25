import logging
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config import TopicConfig
from agent.nodes.augmenter import make_augmenter
from agent.nodes.clarifier import make_clarifier
from agent.nodes.decomposer import decompose
from agent.nodes.formatter import format_response
from agent.nodes.grader import grade
from agent.nodes.reasoner import make_reasoner
from agent.nodes.retriever import make_retriever
from agent.nodes.screen import screen
from agent.serving import LlmCall
from agent.state import AgentState
from db.repositories.documents import DocumentRepository
from db.repositories.query_log import QueryLogEntry

log = logging.getLogger(__name__)


def _increment_retry(state: AgentState) -> dict:
    next_count = state["retry_count"] + 1
    log.info("retry: attempt %d → augmenting sub_questions before next retrieval", next_count)
    return {"retry_count": next_count}


def _after_grade(state: AgentState) -> str:
    graded = len(state["graded_chunks"])
    retries = state["retry_count"]
    if graded >= 2:
        log.info("route: → reason (%d graded chunks, threshold met)", graded)
        return "reason"
    if retries >= 1:
        if not state["graded_chunks"]:
            log.info("route: → clarify (0 graded chunks after %d retr%s, corpus has nothing relevant)", retries, "y" if retries == 1 else "ies")
            return "clarify"
        log.info("route: → reason (%d graded chunk(s), retry limit reached, proceeding with what we have)", graded)
        return "reason"
    log.info("route: → retry (%d graded chunks below threshold=2, retry_count=%d)", graded, retries)
    return "retry"


def _after_screen(state: AgentState) -> str:
    return END if state["block_reason"] else "decompose"


def route_of(state: Mapping[str, Any]) -> str:
    """Which terminal node answered. Mirrors _after_grade: clarify runs only when nothing passed grading."""
    if block := state.get("block_reason"):
        return "blocked_input" if block.startswith("input check") else "blocked_output"
    return "reason" if state.get("graded_chunks") else "clarify"


def log_entry(run_id: str, question: str, state: Mapping[str, Any], latency_ms: int,
              conversation_id: str | None = None, model: str | None = None,
              route: str | None = None, reason: str | None = None, llm_calls: list[LlmCall] | None = None,
              streamed: str = "") -> QueryLogEntry:
    """The query_log row for a run; every /ask request is recorded through this.

    The state may be partial (a timed-out or failed run), so every key has a default; pass route and
    reason to override what the state says. streamed is the answer text already sent to the client: it is what
    the user saw, so it is logged instead of a later replacement by the output check.
    """
    graded = state.get("graded_chunks", [])
    sub_questions = state.get("sub_questions", [])
    return QueryLogEntry(
        run_id=run_id,
        question=question,
        answer=streamed or state.get("answer", ""),
        # The netloc of the top passage's source
        domain=graded[0].metadata.get("source") if graded else None,
        latency_ms=latency_ms,
        route=route or route_of(state),
        sub_question_count=len(sub_questions),
        retrieved_chunk_count=len(state.get("retrieved_chunks", [])),
        graded_chunk_count=len(graded),
        retry_count=state.get("retry_count", 0),
        citations=state.get("citations", []),
        graded_chunk_ids=[c.id for c in graded],
        conversation_id=conversation_id,
        model=model,
        sub_questions=sub_questions,
        reason=reason or state.get("block_reason") or None,
        llm_calls=[asdict(c) for c in llm_calls or []],
        streamed=bool(streamed),
        lang=state.get("lang"),
    )


def build_graph(config: TopicConfig, repo: DocumentRepository) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("screen", screen)
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve", make_retriever(repo))
    graph.add_node("grade", grade)
    graph.add_node("retry", _increment_retry)
    graph.add_node("augment", make_augmenter())
    graph.add_node("reason", make_reasoner(config))
    graph.add_node("clarify", make_clarifier(config, repo))
    graph.add_node("format", format_response)

    graph.set_entry_point("screen")
    graph.add_conditional_edges("screen", _after_screen, {END: END, "decompose": "decompose"})
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _after_grade,
        {"reason": "reason", "retry": "retry", "clarify": "clarify"},
    )
    graph.add_edge("retry", "augment")
    graph.add_edge("augment", "retrieve")
    graph.add_edge("reason", "format")
    graph.add_edge("clarify", END)

    return graph.compile()
