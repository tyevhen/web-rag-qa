from typing import cast

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent.config import TopicConfig
from agent.graph import _after_grade, build_graph, route_of
from agent.nodes import _llm
from agent.state import AgentState, Chunk
from db.repository import DocumentRepository


def _state(graded: int = 0, retry_count: int = 0) -> AgentState:
    chunks = [Chunk(content=f"c{i}", metadata={}, id=str(i), score=0.0) for i in range(graded)]
    return {"question": "q", "history": [], "sub_questions": [], "retrieved_chunks": chunks,
            "graded_chunks": chunks, "answer": "", "citations": [], "clarifying_questions": [], "retry_count": retry_count}


@pytest.mark.parametrize(("graded", "retries", "expected"), [
    (2, 0, "reason"),   # enough evidence
    (1, 0, "retry"),    # too little, retry left
    (0, 0, "retry"),
    (1, 1, "reason"),   # best effort after retry
    (0, 1, "clarify"),  # nothing after retry
])
def test_after_grade(graded: int, retries: int, expected: str):
    assert _after_grade(_state(graded, retries)) == expected


class EmptyRepo:
    async def similarity_search(self, embedding: list[float], k: int) -> list[dict]:
        return []

    async def fulltext_search(self, query: str, k: int) -> list[dict]:
        return []


async def test_empty_clarifier_output_is_still_a_clarify_route(monkeypatch):
    # decompose, augment, clarify — the clarifier returns nothing, as gpt-oss does when it runs out of tokens
    llm = FakeListChatModel(responses=["sub question", "plain sub question", ""])
    monkeypatch.setattr(_llm, "get_groq_llm", lambda: llm)
    monkeypatch.setattr("agent.nodes.retriever._embed_documents", lambda texts: [[0.0] for _ in texts])
    graph = build_graph(TopicConfig(system_prompt="test"), cast(DocumentRepository, EmptyRepo()))

    state = await graph.ainvoke(_state())

    assert state["retry_count"] == 1
    assert state["clarifying_questions"] == []
    assert route_of(state) == "clarify"
