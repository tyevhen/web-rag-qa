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
    citations: list[dict]
    graded_chunk_ids: list[str]
    conversation_id: str | None = None  # None only for rows without a conversation (rejected requests)
    model: str | None = None
    sub_questions: list[str] = field(default_factory=list)
    reason: str | None = None  # why a request was blocked, timed out, failed or rejected
    llm_calls: list[dict] = field(default_factory=list)  # serving numbers per LLM call (agent/serving.py LlmCall)
    streamed: bool = False  # the answer was sent to the client as it was written (/ask/stream)
    lang: str | None = None  # the question's language, which retrieval searched (agent/lang.py)


class QueryLogRepository:
    """The query_log table: one row per /ask request (any outcome), written by /ask and read by the dashboard and nightly judge."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def log_query(self, entry: QueryLogEntry) -> None:
        await self._pool.execute(
            """
            INSERT INTO query_log (
                run_id, question, answer, domain, latency_ms, route,
                sub_question_count, retrieved_chunk_count, graded_chunk_count,
                retry_count, citations, graded_chunk_ids, conversation_id, model, sub_questions, reason, llm_calls, streamed, lang
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            """,
            uuid.UUID(entry.run_id),
            entry.question[:4000],  # rejected requests can carry arbitrarily long text
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
            uuid.UUID(entry.conversation_id) if entry.conversation_id else None,
            entry.model,
            entry.sub_questions,
            entry.reason,
            entry.llm_calls,
            entry.streamed,
            entry.lang,
        )

    async def query_log_since(self, days: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT created_at, question, route, latency_ms, retry_count,
                   retrieved_chunk_count, graded_chunk_count, llm_calls, streamed
            FROM query_log
            WHERE created_at > now() - make_interval(days => $1)
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
            SELECT conversation_id::text AS chat_id, (array_agg(question ORDER BY created_at))[1] AS title,
                   min(created_at) AS started, array_agg(DISTINCT model) AS models, count(*) AS turns
            FROM query_log
            WHERE conversation_id IS NOT NULL
            GROUP BY conversation_id
            ORDER BY max(created_at) DESC, conversation_id
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )

    async def chat_turns(self, chat_id: str) -> list[asyncpg.Record]:
        """Every recorded turn of one chat, oldest first, with any judge scores as {metric: score}."""
        return await self._pool.fetch(
            """
            SELECT q.run_id::text, q.model, q.question, q.answer, q.route, q.reason, q.latency_ms, q.retry_count, q.llm_calls,
                   q.sub_questions, q.retrieved_chunk_count, q.graded_chunk_count, q.citations,
                   q.graded_chunk_ids::text[] AS graded_chunk_ids, q.created_at, s.scores
            FROM query_log q
            LEFT JOIN LATERAL (
                SELECT jsonb_object_agg(metric, score) AS scores FROM query_eval e WHERE e.run_id = q.run_id
            ) s ON true
            WHERE q.conversation_id = $1
            ORDER BY q.created_at
            """,
            uuid.UUID(chat_id),
        )
