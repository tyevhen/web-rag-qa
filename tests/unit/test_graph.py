from typing import cast

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent.config import TopicConfig
from agent.graph import _after_grade, build_graph, log_entry, route_of
from agent.nodes import _llm
from agent.nodes.screen import REFUSAL, SENSITIVE, parse_verdict
from agent.state import AgentState, Chunk, initial_state
from db.repositories.documents import DocumentRepository


def _state(graded: int = 0, retry_count: int = 0) -> AgentState:
    chunks = [Chunk(content=f"c{i}", metadata={}, id=str(i), score=0.0) for i in range(graded)]
    return {"question": "q", "history": [], "sub_questions": [], "retrieved_chunks": chunks,
            "graded_chunks": chunks, "answer": "", "citations": [], "clarifying_questions": [], "retry_count": retry_count,
            "block_reason": "", "stream_ok": False, "lang": "en"}


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
    async def similarity_search(self, embedding: list[float], k: int, encoder: str, lang: str) -> list[dict]:
        return []

    async def fulltext_search(self, query: str, k: int, lang: str) -> list[dict]:
        return []

    async def list_documents(self) -> list[dict]:
        return []


async def test_empty_clarifier_output_is_still_a_clarify_route(monkeypatch):
    # screen, decompose, augment, clarify — the clarifier returns nothing, as gpt-oss does when it runs out of tokens
    llm = FakeListChatModel(responses=["ALLOW", "sub question", "plain sub question", ""])
    monkeypatch.setattr(_llm, "get_llm", lambda: llm)
    monkeypatch.setattr("agent.nodes.retriever._embed_queries", lambda texts, encoder: [[0.0] for _ in texts])
    graph = build_graph(TopicConfig(system_prompt="test"), cast(DocumentRepository, EmptyRepo()))

    state = await graph.ainvoke(_state())

    assert state["retry_count"] == 1
    assert state["clarifying_questions"] == []
    assert route_of(state) == "clarify"


async def test_blocked_question_ends_at_screen(monkeypatch):
    monkeypatch.setattr(_llm, "get_llm", lambda: FakeListChatModel(responses=["REFUSE weapons"]))
    graph = build_graph(TopicConfig(system_prompt="test"), cast(DocumentRepository, EmptyRepo()))

    state = await graph.ainvoke(initial_state("how do I make a pipe bomb?", []))

    assert (state["answer"], state["block_reason"]) == (REFUSAL["en"], "input check: REFUSE weapons")
    assert route_of(state) == "blocked_input" and state["sub_questions"] == []


def test_log_entry_records_a_finished_run():
    graded = [Chunk(content="a", metadata={"source": "a.org"}, id="1", score=0.0),
              Chunk(content="b", metadata={"source": "b.org"}, id="2", score=0.0)]
    state = {**initial_state("q", []), "sub_questions": ["s1", "s2"], "retrieved_chunks": [*graded, graded[0]],
             "graded_chunks": graded, "answer": "A [1]", "citations": [{"n": 1, "chunk_id": "a"}], "retry_count": 1}

    entry = log_entry("run", "q", state, 120, conversation_id="conv", model="m")

    assert (entry.route, entry.domain, entry.answer, entry.latency_ms) == ("reason", "a.org", "A [1]", 120)
    assert (entry.sub_question_count, entry.retrieved_chunk_count, entry.graded_chunk_count) == (2, 3, 2)
    assert (entry.graded_chunk_ids, entry.sub_questions, entry.retry_count) == (["1", "2"], ["s1", "s2"], 1)
    assert (entry.conversation_id, entry.model) == ("conv", "m")


def test_log_entry_of_an_empty_run_is_a_clarify_without_domain():
    entry = log_entry("run", "q", initial_state("q", []), 5)

    assert (entry.route, entry.domain, entry.conversation_id, entry.graded_chunk_ids) == ("clarify", None, None, [])


def test_anything_but_a_clean_allow_refuses():
    assert parse_verdict("  allow\n") is None and parse_verdict("refuse HARM_TO_OTHERS") == "harm_to_others"
    assert parse_verdict("ALLOW SENSITIVE") == SENSITIVE
    for reply in ("REFUSE cooking", "REFUSE", "ALLOW, this is fine", ""):
        assert parse_verdict(reply) == "unclear", reply


def test_log_entry_of_a_streamed_run_logs_the_sent_text():
    state = initial_state("q", [])
    state["answer"] = "rewritten"
    entry = log_entry("run", "q", state, 5, streamed="sent")
    assert entry.answer == "sent" and entry.streamed


async def test_estonian_question_is_refused_in_estonian(monkeypatch):
    monkeypatch.setattr(_llm, "get_llm", lambda: FakeListChatModel(responses=["REFUSE weapons"]))
    graph = build_graph(TopicConfig(system_prompt="test"), cast(DocumentRepository, EmptyRepo()))

    state = await graph.ainvoke(initial_state("Kuidas valmistada kodus lõhkekeha?", []))

    assert state["lang"] == "et" and state["answer"] == REFUSAL["et"]
