import logging

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config import TopicConfig
from agent.nodes.augmenter import make_augmenter
from agent.nodes.clarifier import make_clarifier
from agent.nodes.decomposer import make_decomposer
from agent.nodes.formatter import format_response
from agent.nodes.grader import grade
from agent.nodes.reasoner import make_reasoner
from agent.nodes.retriever import make_retriever
from agent.state import AgentState
from db.repository import DocumentRepository

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


def build_graph(config: TopicConfig, repo: DocumentRepository) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("decompose", make_decomposer(config))
    graph.add_node("retrieve", make_retriever(repo))
    graph.add_node("grade", grade)
    graph.add_node("retry", _increment_retry)
    graph.add_node("augment", make_augmenter())
    graph.add_node("reason", make_reasoner(config))
    graph.add_node("clarify", make_clarifier(config))
    graph.add_node("format", format_response)

    graph.set_entry_point("decompose")
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
