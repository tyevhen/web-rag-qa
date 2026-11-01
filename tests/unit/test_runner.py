from typing import Any, cast

import pytest
from langgraph.graph.state import CompiledStateGraph

from agent.state import Chunk
from db.repository import DocumentRepository
from tests.eval import runner
from tests.eval.dataset import GoldenRow, content_hash


def _chunk(text: str) -> Chunk:
    return Chunk(content=text, metadata={}, id=text, score=0.0)


class ScriptedGraph:
    """Replays the stream of a run that retried: the first retrieval found `first`, the one after augmenting found 'b'."""

    def __init__(self, first: list[str]) -> None:
        self.first = first

    async def astream(self, state: Any, stream_mode: list[str]):
        yield "updates", {"retrieve": {"retrieved_chunks": [_chunk(t) for t in self.first]}}
        yield "updates", {"retrieve": {"retrieved_chunks": [_chunk("b")]}}
        yield "values", {"retrieved_chunks": [_chunk("b")], "graded_chunks": [], "sub_questions": [],
                         "answer": "", "retry_count": 1}


@pytest.mark.parametrize(("first", "expected"), [(["a"], [content_hash("a")]), ([], [])])
async def test_first_pass_retrieval_is_scored_separately_from_the_final_one(monkeypatch, first, expected):
    async def no_ablation(repo, row):
        return {}

    monkeypatch.setattr(runner, "_ablation", no_ablation)
    monkeypatch.setattr(runner, "score_chunks", lambda queries, chunks: [0.0] * len(chunks))

    result = await runner._run_one(cast(CompiledStateGraph, ScriptedGraph(first)), cast(DocumentRepository, None),
                                   GoldenRow(question="q", category="answerable"))

    assert [c.content for c in result.retrieved] == ["b"]
    assert result.ablation == {"hybrid_decomposed_first_pass": expected}  # an empty first pass stays empty
    assert result.route == "clarify"
