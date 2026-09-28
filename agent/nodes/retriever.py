import logging

from agent.embed import embed_query as _embed_query
from agent.state import AgentState, Chunk
from db.repository import DocumentRepository

log = logging.getLogger(__name__)

_TOP_K = 10
_FINAL_K = 6
_RRF_K = 60


def _rrf(scores: dict[str, float], ranked_ids: list[str]) -> None:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank)


def make_retriever(repo: DocumentRepository):
    async def retrieve(state: AgentState) -> dict:
        queries = state["sub_questions"] or [state["question"]]
        chunks: dict[str, Chunk] = {}
        rrf_scores: dict[str, float] = {}

        log.info("retrieve: retry_count=%d, %d queries: %s", state["retry_count"], len(queries), queries)

        for i, query in enumerate(queries, start=1):
            # Dense leg — pgvector cosine similarity
            query_embedding = _embed_query(query)
            dense_rows = await repo.similarity_search(query_embedding, _TOP_K)
            dense_ids: list[str] = []
            for row in dense_rows:
                cid = row["id"]
                chunks.setdefault(cid, Chunk(content=row["content"], metadata=dict(row["metadata"] or {}), id=cid, score=0.0))
                dense_ids.append(cid)

            # Sparse leg — PostgreSQL full-text search
            sparse_rows = await repo.fulltext_search(query, _TOP_K)
            sparse_ids: list[str] = []
            for row in sparse_rows:
                cid = row["id"]
                chunks.setdefault(cid, Chunk(content=row["content"], metadata=dict(row["metadata"] or {}), id=cid, score=0.0))
                sparse_ids.append(cid)

            log.info("retrieve: query %d/%d %r → dense=%d sparse=%d", i, len(queries), query, len(dense_ids), len(sparse_ids))

            _rrf(rrf_scores, dense_ids)
            _rrf(rrf_scores, sparse_ids)

        if not chunks:
            log.info("retrieve: no chunks found")
            return {"retrieved_chunks": []}

        for cid, score in rrf_scores.items():
            chunks[cid].score = score

        top = sorted(chunks.values(), key=lambda c: c.score, reverse=True)[:_FINAL_K]
        log.info("retrieve: returning %d chunks after RRF (top score=%.4f)", len(top), top[0].score if top else 0)
        return {"retrieved_chunks": top}

    return retrieve
