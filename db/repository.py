import uuid
from dataclasses import dataclass

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


class DocumentRepository:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def create_document(self, filename: str, domain: str) -> str:
        return str(
            await self._pool.fetchval(
                "INSERT INTO documents (filename, domain) VALUES ($1, $2) RETURNING id",
                filename,
                domain,
            )
        )

    async def insert_chunks(
        self,
        doc_id: str,
        contents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        await self._pool.executemany(
            "INSERT INTO chunks (document_id, content, metadata, embedding) VALUES ($1, $2, $3, $4)",
            [
                (uuid.UUID(doc_id), content, metadata, embedding)
                for content, metadata, embedding in zip(contents, metadatas, embeddings)
            ],
        )

    async def similarity_search(self, embedding: list[float], k: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT id::text, content, metadata,
                   1 - (embedding <=> $1) AS score
            FROM chunks
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            embedding,
            k,
        )

    async def fulltext_search(self, query: str, k: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT id::text, content, metadata
            FROM chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
            ORDER BY ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) DESC
            LIMIT $2
            """,
            query,
            k,
        )

    async def list_documents(self) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT d.id::text, d.filename, d.domain, d.uploaded_at,
                   COUNT(c.id) AS chunks
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id, d.filename, d.domain, d.uploaded_at
            ORDER BY d.uploaded_at DESC
            """
        )

    async def list_sections(self) -> list[str]:
        rows = await self._pool.fetch(
            """
            SELECT DISTINCT metadata->>'section' AS section
            FROM chunks
            WHERE metadata->>'section' IS NOT NULL
              AND metadata->>'section' != ''
            ORDER BY section
            """
        )
        return [row["section"] for row in rows]

    async def log_query(self, entry: QueryLogEntry) -> None:
        await self._pool.execute(
            """
            INSERT INTO query_log (
                run_id, question, answer, domain, latency_ms, route,
                sub_question_count, retrieved_chunk_count, graded_chunk_count,
                retry_count, citations
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
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
        )
