import uuid

import asyncpg
from asyncpg import Pool


class EvalQueriesMixin:
    """Evaluation SQL, mixed into DocumentRepository to keep repository.py under the line limit."""

    _pool: Pool

    async def sample_chunks(self, n: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            "SELECT id::text, content, metadata FROM chunks ORDER BY random() LIMIT $1",
            n,
        )

    async def get_chunks_by_ids(self, ids: list[str]) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            "SELECT id::text, content, metadata FROM chunks WHERE id = ANY($1::uuid[])",
            [uuid.UUID(i) for i in ids],
        )

    async def fetch_unscored_queries(self, limit: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT q.run_id::text, q.question, q.answer, q.graded_chunk_ids::text[] AS graded_chunk_ids
            FROM query_log q
            WHERE q.route = 'reason'
              AND q.run_id IS NOT NULL
              AND cardinality(q.graded_chunk_ids) > 0
              AND q.created_at > now() - interval '7 days'
              AND NOT EXISTS (SELECT 1 FROM query_eval e WHERE e.run_id = q.run_id)
            ORDER BY random()
            LIMIT $1
            """,
            limit,
        )

    async def save_eval_scores(
        self, run_id: str, scores: dict[str, float], judge_model: str
    ) -> None:
        await self._pool.executemany(
            "INSERT INTO query_eval (run_id, metric, score, judge_model) VALUES ($1, $2, $3, $4)",
            [(uuid.UUID(run_id), metric, score, judge_model) for metric, score in scores.items()],
        )

    async def recent_eval_means(self, days: int) -> dict[str, float]:
        rows = await self._pool.fetch(
            """
            SELECT metric, AVG(score) AS mean
            FROM query_eval
            WHERE evaluated_at > now() - make_interval(days => $1) AND score IS NOT NULL
            GROUP BY metric
            """,
            days,
        )
        return {row["metric"]: float(row["mean"]) for row in rows}
