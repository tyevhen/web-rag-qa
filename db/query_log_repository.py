import uuid
from dataclasses import dataclass, field

import asyncpg
from asyncpg import Pool


@dataclass
class QueryLogEntry:
    run_id: str
    question: str
    answer: str
    domain: str | None
    latency_ms: int
    route: str
    sub_question_count: int
    retrieved_chunk_count: int
    graded_chunk_count: int
    retry_count: int
    citations: list[str]
    graded_chunk_ids: list[str]
    chat_id: str | None = None  # set by the dashboard Chat page; None for API requests
    model: str | None = None
    sub_questions: list[str] = field(default_factory=list)


class QueryLogRepository:
    """The query_log table: one row per answered request, written by /ask and read by the dashboard and nightly judge."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def log_query(self, entry: QueryLogEntry) -> None:
        await self._pool.execute(
            """
            INSERT INTO query_log (
                run_id, question, answer, domain, latency_ms, route,
                sub_question_count, retrieved_chunk_count, graded_chunk_count,
                retry_count, citations, graded_chunk_ids, chat_id, model, sub_questions
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """,
            uuid.UUID(entry.run_id),
            entry.question,
            entry.answer,
            entry.domain,
            entry.latency_ms,
            entry.route,
            entry.sub_question_count,
            entry.retrieved_chunk_count,
            entry.graded_chunk_count,
            entry.retry_count,
            entry.citations,
            [uuid.UUID(i) for i in entry.graded_chunk_ids],
            uuid.UUID(entry.chat_id) if entry.chat_id else None,
            entry.model,
            entry.sub_questions,
        )

    async def query_log_since(self, days: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT created_at, question, route, latency_ms, retry_count,
                   retrieved_chunk_count, graded_chunk_count
            FROM query_log
            WHERE created_at > now() - make_interval(days => $1) AND chat_id IS NULL
            ORDER BY created_at
            """,
            days,
        )

    async def fetch_unscored_queries(self, limit: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT q.run_id::text, q.question, q.answer, q.graded_chunk_ids::text[] AS graded_chunk_ids
            FROM query_log q
            WHERE q.route = 'reason'
              AND q.run_id IS NOT NULL
              AND q.chat_id IS NULL
              AND cardinality(q.graded_chunk_ids) > 0
              AND q.created_at > now() - interval '7 days'
              AND NOT EXISTS (SELECT 1 FROM query_eval e WHERE e.run_id = q.run_id)
            ORDER BY random()
            LIMIT $1
            """,
            limit,
        )

    async def query_for_scoring(self, run_id: str) -> asyncpg.Record | None:
        return await self._pool.fetchrow(
            """
            SELECT run_id::text, question, answer, graded_chunk_ids::text[] AS graded_chunk_ids
            FROM query_log WHERE run_id = $1
            """,
            uuid.UUID(run_id),
        )

    async def recent_chats(self, limit: int, offset: int = 0) -> list[asyncpg.Record]:
        """One row per chat, most recently active first: first question as title, start time, models, answers."""
        return await self._pool.fetch(
            """
            SELECT chat_id::text, (array_agg(question ORDER BY created_at))[1] AS title,
                   min(created_at) AS started, array_agg(DISTINCT model) AS models, count(*) AS turns
            FROM query_log
            WHERE chat_id IS NOT NULL
            GROUP BY chat_id
            ORDER BY max(created_at) DESC, chat_id
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )

    async def chat_turns(self, chat_id: str) -> list[asyncpg.Record]:
        """Every recorded turn of one chat, oldest first, with any judge scores as {metric: score}."""
        return await self._pool.fetch(
            """
            SELECT q.run_id::text, q.model, q.question, q.answer, q.route, q.latency_ms, q.retry_count,
                   q.sub_questions, q.retrieved_chunk_count, q.graded_chunk_count, q.citations,
                   q.graded_chunk_ids::text[] AS graded_chunk_ids, q.created_at, s.scores
            FROM query_log q
            LEFT JOIN LATERAL (
                SELECT jsonb_object_agg(metric, score) AS scores FROM query_eval e WHERE e.run_id = q.run_id
            ) s ON true
            WHERE q.chat_id = $1
            ORDER BY q.created_at
            """,
            uuid.UUID(chat_id),
        )
