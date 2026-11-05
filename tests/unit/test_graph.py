from typing import cast

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent.config import TopicConfig
from agent.graph import _after_grade, build_graph, log_entry, route_of
from agent.nodes import _llm
from agent.state import AgentState, Chunk, initial_state
from db.repositories.documents import DocumentRepository


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
    async def similarity_search(self, embedding: list[float], k: int, encoder: str) -> list[dict]:
        return []

    async def fulltext_search(self, query: str, k: int) -> list[dict]:
        return []


async def test_empty_clarifier_output_is_still_a_clarify_route(monkeypatch):
    # decompose, augment, clarify — the clarifier returns nothing, as gpt-oss does when it runs out of tokens
    llm = FakeListChatModel(responses=["sub question", "plain sub question", ""])
    monkeypatch.setattr(_llm, "get_groq_llm", lambda: llm)
    monkeypatch.setattr("agent.nodes.retriever._embed_queries", lambda texts, encoder: [[0.0] for _ in texts])
    graph = build_graph(TopicConfig(system_prompt="test"), cast(DocumentRepository, EmptyRepo()))

    state = await graph.ainvoke(_state())

    assert state["retry_count"] == 1
    assert state["clarifying_questions"] == []
    assert route_of(state) == "clarify"


def test_log_entry_records_a_finished_run():
    graded = [Chunk(content="a", metadata={"source": "a.org"}, id="1", score=0.0),
              Chunk(content="b", metadata={"source": "b.org"}, id="2", score=0.0)]
    state = {**initial_state("q", []), "sub_questions": ["s1", "s2"], "retrieved_chunks": [*graded, graded[0]],
             "graded_chunks": graded, "answer": "A [1]", "citations": ["[1] a.org"], "retry_count": 1}

    entry = log_entry("run", "q", state, 120, chat_id="chat", model="m")

    assert (entry.route, entry.domain, entry.answer, entry.latency_ms) == ("reason", "a.org", "A [1]", 120)
    assert (entry.sub_question_count, entry.retrieved_chunk_count, entry.graded_chunk_count) == (2, 3, 2)
    assert (entry.graded_chunk_ids, entry.sub_questions, entry.retry_count) == (["1", "2"], ["s1", "s2"], 1)
    assert (entry.chat_id, entry.model) == ("chat", "m")


def test_log_entry_of_an_empty_run_is_a_clarify_without_domain():
    entry = log_entry("run", "q", initial_state("q", []), 5)

    assert (entry.route, entry.domain, entry.chat_id, entry.graded_chunk_ids) == ("clarify", None, None, [])
