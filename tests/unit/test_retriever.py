import threading
from typing import cast

from agent.nodes import retriever
from agent.state import AgentState
from db.repository import DocumentRepository


class FakeRepo:
    """Dense and sparse legs return fixed rankings, independent of the query."""

    def __init__(self, dense: list[str], sparse: list[str]) -> None:
        self.dense, self.sparse = dense, sparse

    def as_repo(self) -> DocumentRepository:
        return cast(DocumentRepository, self)

    async def similarity_search(self, embedding: list[float], k: int, encoder: str) -> list[dict]:
        return [{"id": i, "content": i, "metadata": {}} for i in self.dense[:k]]

    async def fulltext_search(self, query: str, k: int) -> list[dict]:
        return [{"id": i, "content": i, "metadata": {}} for i in self.sparse[:k]]


def _state(question: str, sub_questions: list[str]) -> AgentState:
    return {"question": question, "history": [], "sub_questions": sub_questions, "retrieved_chunks": [],
            "graded_chunks": [], "answer": "", "citations": [], "clarifying_questions": [], "retry_count": 0}


async def test_embeds_off_the_event_loop_once_per_unique_query(monkeypatch):
    calls: list[tuple[list[str], bool]] = []

    def fake_embed(texts: list[str], encoder: object) -> list[list[float]]:
        calls.append((texts, threading.current_thread() is threading.main_thread()))
        return [[0.0] for _ in texts]

    monkeypatch.setattr(retriever, "_embed_queries", fake_embed)
    await retriever.make_retriever(FakeRepo(["a"], []).as_repo())(_state("q", ["q", "s1"]))

    assert calls == [(["q", "s1"], False)]


async def test_rrf_ranks_chunks_found_by_both_legs_first(monkeypatch):
    monkeypatch.setattr(retriever, "_embed_queries", lambda texts, encoder: [[0.0] for _ in texts])
    repo = FakeRepo(dense=["a", "b", "c"], sparse=["c", "d"])
    out = await retriever.make_retriever(repo.as_repo())(_state("q", []))

    ids = [c.id for c in out["retrieved_chunks"]]
    assert ids[0] == "c"  # rank 3 dense + rank 1 sparse beats rank 1 in one leg only
    assert set(ids) == {"a", "b", "c", "d"}
