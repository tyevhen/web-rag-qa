import asyncio
import logging
from typing import Any

from sentence_transformers import CrossEncoder

from agent.state import AgentState, Chunk

log = logging.getLogger(__name__)

_THRESHOLD = 0.0
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        log.info("grader: loading cross-encoder %s", _MODEL_NAME)
        _model = CrossEncoder(_MODEL_NAME)
    return _model


async def grade(state: AgentState) -> dict[str, Any]:
    chunks = state["retrieved_chunks"]
    question = state["question"]

    if not chunks:
        log.info("grade: no chunks to score")
        return {"graded_chunks": []}

    log.info("grade: scoring %d chunks (retry_count=%d)", len(chunks), state["retry_count"])

    pairs = [(question, chunk.content[:500]) for chunk in chunks]
    scores = await asyncio.to_thread(_get_model().predict, pairs)

    graded: list[Chunk] = []
    for chunk, score in zip(chunks, scores):
        verdict = "relevant" if score >= _THRESHOLD else "not relevant"
        log.info("grade: score=%.3f → %s | %.80r", score, verdict, chunk.content)
        if score >= _THRESHOLD:
            graded.append(chunk)

    log.info("grade: %d/%d chunks passed threshold=%.1f", len(graded), len(chunks), _THRESHOLD)
    return {"graded_chunks": graded}
