from typing import Any, cast

import pytest

from agent.nodes import formatter
from agent.nodes.formatter import format_response
from agent.nodes.screen import REFUSAL
from agent.state import AgentState, Chunk


def _state(answer: str) -> AgentState:
    chunks = [Chunk(content=t, metadata={"url": f"https://x/{t}"}, id=t, score=0.0) for t in ("a", "b", "c")]
    state: dict[str, Any] = {"graded_chunks": chunks, "answer": answer, "lang": "en"}
    return cast(AgentState, state)


@pytest.fixture(autouse=True)
def allow_all(monkeypatch):
    async def allow(text: str) -> None:
        return None

    monkeypatch.setattr(formatter, "check", allow)


async def test_cited_sources_in_order_of_first_mention_out_of_range_dropped():
    result = await format_response(_state("B [2]. A [1][2]. Bogus [9]."))

    assert result["citations"] == [{"n": 2, "chunk_id": "b", "section": "", "source_url": "https://x/b"},
                                   {"n": 1, "chunk_id": "a", "section": "", "source_url": "https://x/a"}]


async def test_no_markers_means_no_citations():
    assert (await format_response(_state("An answer without markers."))) == {"citations": []}


async def test_flagged_answer_becomes_the_refusal(monkeypatch):
    async def flag(text: str) -> str:
        return "weapons"

    monkeypatch.setattr(formatter, "check", flag)

    result = await format_response(_state("Mix these [1]."))

    assert result == {"answer": REFUSAL["en"], "citations": [], "clarifying_questions": [],
                      "block_reason": "output check on reason: REFUSE weapons"}
