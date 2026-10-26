import asyncpg
from asyncpg import Pool


class DashboardQueriesMixin:
    """Read-only dashboard SQL, mixed into DocumentRepository to keep repository.py under the line limit."""

    _pool: Pool

    async def chunk_embeddings(self) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT c.id::text, c.content, c.embedding, d.filename AS source, d.uploaded_at
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY d.uploaded_at, (c.metadata->>'chunk_index')::int
            """
        )

    async def query_log_since(self, days: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT created_at, question, route, latency_ms, retry_count,
                   retrieved_chunk_count, graded_chunk_count
            FROM query_log
            WHERE created_at > now() - make_interval(days => $1)
            ORDER BY created_at
            """,
            days,
        )

    async def eval_scores_since(self, days: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT e.evaluated_at, e.metric, e.score, e.judge_model, q.question, q.answer
            FROM query_eval e
            JOIN query_log q ON q.run_id = e.run_id
            WHERE e.evaluated_at > now() - make_interval(days => $1)
            ORDER BY e.evaluated_at
            """,
            days,
        )
