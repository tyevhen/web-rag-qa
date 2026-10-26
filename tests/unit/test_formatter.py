from typing import Any, cast

from agent.nodes.formatter import format_response
from agent.state import AgentState, Chunk


def _state(answer: str) -> AgentState:
    chunks = [Chunk(content=t, metadata={"url": f"https://x/{t}"}, id=t, score=0.0) for t in ("a", "b", "c")]
    state: dict[str, Any] = {"graded_chunks": chunks, "answer": answer}
    return cast(AgentState, state)


async def test_cited_sources_in_order_of_first_mention_out_of_range_dropped():
    result = await format_response(_state("B [2]. A [1][2]. Bogus [9]."))

    assert result["citations"] == ["[2] | url: https://x/b", "[1] | url: https://x/a"]


async def test_no_markers_means_no_citations():
    assert (await format_response(_state("An answer without markers."))) == {"citations": []}
