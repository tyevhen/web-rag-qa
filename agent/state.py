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
