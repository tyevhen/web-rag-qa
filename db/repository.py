import uuid

import asyncpg
from asyncpg import Pool

from db.embedding_repository import encoder_literal


class DocumentRepository:
    """Documents and their chunks: ingestion writes, search, and chunk reads for eval and the dashboard."""

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
        embeddings: dict[str, list[list[float]]],
    ) -> None:
        """Chunks and one vector per chunk for each encoder in `embeddings`, all or nothing."""
        ids = [uuid.uuid4() for _ in contents]
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                "INSERT INTO chunks (id, document_id, content, metadata) VALUES ($1, $2, $3, $4)",
                [(i, uuid.UUID(doc_id), c, m) for i, c, m in zip(ids, contents, metadatas)],
            )
            await conn.executemany(
                "INSERT INTO chunk_embeddings (chunk_id, encoder, embedding) VALUES ($1, $2, $3)",
                [(i, encoder, v) for encoder, vectors in embeddings.items() for i, v in zip(ids, vectors)],
            )

    async def similarity_search(self, embedding: list[float], k: int, encoder: str) -> list[asyncpg.Record]:
        # Encoder and size inlined so the query matches that encoder's partial index (db/embedding_repository.py)
        vec = f"e.embedding::vector({len(embedding)})"
        return await self._pool.fetch(
            f"""
            SELECT c.id::text, c.content, c.metadata, 1 - ({vec} <=> $1) AS score
            FROM chunk_embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            WHERE e.encoder = {encoder_literal(encoder)}
            ORDER BY {vec} <=> $1
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

    async def chunk_embeddings(self, encoder: str) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            """
            SELECT c.id::text, c.content, e.embedding, d.filename AS source, d.uploaded_at
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN chunk_embeddings e ON e.chunk_id = c.id AND e.encoder = $1
            ORDER BY d.uploaded_at, (c.metadata->>'chunk_index')::int
            """,
            encoder,
        )
