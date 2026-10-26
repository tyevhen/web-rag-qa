import asyncio
import logging
from typing import Any

from sentence_transformers import CrossEncoder

from agent.state import AgentState, Chunk

log = logging.getLogger(__name__)

# Logit cutoff, tuned on the golden set with whole-chunk scoring: on-topic rows keep a chunk at any value in
# [-2.0, -0.5] (15/18), while the best off-topic chunk scores <= -2.03. -1.0 keeps a 1-point margin from it;
# 0.0 dropped relevant evidence (14/18, and fewer rows reached the 2 chunks needed to answer without retry).
_THRESHOLD = -1.0
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        log.info("grader: loading cross-encoder %s", _MODEL_NAME)
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def grading_queries(question: str, sub_questions: list[str]) -> list[str]:
    """Original question plus the decomposer's standalone sub-questions (they resolve history references)."""
    return list(dict.fromkeys([question, *sub_questions]))


def score_chunks(queries: list[str], chunks: list[Chunk]) -> list[float]:
    """Best cross-encoder logit per chunk across all queries — a chunk answering one sub-question is relevant."""
    # Whole chunk: chunks are capped at 1500 chars (~400 tokens), within the model's 512-token window
    pairs = [(q, chunk.content) for chunk in chunks for q in queries]
    flat = _get_model().predict(pairs)
    n = len(queries)
    return [float(max(flat[i * n : (i + 1) * n])) for i in range(len(chunks))]


async def grade(state: AgentState) -> dict[str, Any]:
    chunks = state["retrieved_chunks"]

    if not chunks:
        log.info("grade: no chunks to score")
        return {"graded_chunks": []}

    queries = grading_queries(state["question"], state["sub_questions"])
    log.info("grade: scoring %d chunks against %d queries (retry_count=%d)", len(chunks), len(queries), state["retry_count"])

    scores = await asyncio.to_thread(score_chunks, queries, chunks)

    graded: list[Chunk] = []
    for chunk, score in zip(chunks, scores):
        verdict = "relevant" if score >= _THRESHOLD else "not relevant"
        log.info("grade: score=%.3f → %s | %.80r", score, verdict, chunk.content)
        if score >= _THRESHOLD:
            graded.append(chunk)

    log.info("grade: %d/%d chunks passed threshold=%.1f", len(graded), len(chunks), _THRESHOLD)
    return {"graded_chunks": graded}
