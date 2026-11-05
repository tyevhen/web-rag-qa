from dataclasses import dataclass
from typing import TypedDict


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
    citations: list[str]
    clarifying_questions: list[str]
    retry_count: int


def initial_state(question: str, history: list[dict], sub_questions: list[str] | None = None) -> AgentState:
    """A fresh run: the only way callers (API, dashboard Chat, eval runner) start the graph."""
    return {"question": question, "history": history, "sub_questions": sub_questions or [], "retrieved_chunks": [],
            "graded_chunks": [], "answer": "", "citations": [], "clarifying_questions": [], "retry_count": 0}
