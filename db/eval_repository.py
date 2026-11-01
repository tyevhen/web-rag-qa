import uuid

import asyncpg
from asyncpg import Pool


class EvalRepository:
    """The query_eval table: judge scores per answered request (run_id), with the answer joined in for display."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

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
            SELECT e.metric, AVG(e.score) AS mean
            FROM query_eval e
            JOIN query_log q ON q.run_id = e.run_id
            WHERE e.evaluated_at > now() - make_interval(days => $1) AND e.score IS NOT NULL
              AND q.chat_id IS NULL
            GROUP BY e.metric
            """,
            days,
        )
        return {row["metric"]: float(row["mean"]) for row in rows}

    async def eval_scores_since(self, days: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT e.evaluated_at, e.metric, e.score, e.judge_model, q.question, q.answer
            FROM query_eval e
            JOIN query_log q ON q.run_id = e.run_id
            WHERE e.evaluated_at > now() - make_interval(days => $1) AND q.chat_id IS NULL
            ORDER BY e.evaluated_at
            """,
            days,
        )
