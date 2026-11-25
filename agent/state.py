from dataclasses import dataclass
from typing import TypedDict

from agent.lang import question_lang


@dataclass
class Chunk:
    content: str
    metadata: dict
    id: str
    score: float


class AgentState(TypedDict):
    question: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    sub_questions: list[str]
    retrieved_chunks: list[Chunk]
    graded_chunks: list[Chunk]
    answer: str
    citations: list[dict]  # [{"n", "chunk_id", "section", "source_url"}] for the [n] the answer cites
    clarifying_questions: list[str]
    retry_count: int
    block_reason: str  # set by a safety check, e.g. "input check: REFUSE weapons"; "" when not blocked
    stream_ok: bool  # only a clean ALLOW from screen sets it; then /ask/stream may send the answer as it is written
    lang: str  # "en" | "et": retrieval searches only passages in this language


def initial_state(question: str, history: list[dict], sub_questions: list[str] | None = None) -> AgentState:
    """A fresh run: the only way callers (API, dashboard Chat, eval runner) start the graph."""
    return {"question": question, "history": history, "sub_questions": sub_questions or [], "retrieved_chunks": [],
            "graded_chunks": [], "answer": "", "citations": [], "clarifying_questions": [], "retry_count": 0,
            "block_reason": "", "stream_ok": False,
            "lang": question_lang(question)}
