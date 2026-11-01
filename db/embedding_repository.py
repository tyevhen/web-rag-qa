import re
import uuid

import asyncpg
from asyncpg import Pool

_SAFE_NAME = re.compile(r"[\w./-]+")


def encoder_literal(name: str) -> str:
    """The encoder name as an SQL literal. Inlined, not bound: with a bind parameter the planner
    can't prove the per-encoder partial index applies, and search falls back to a full scan."""
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"Unsafe encoder name {name!r}")
    return f"'{name}'"


def _index_name(name: str) -> str:
    return f"emb_{re.sub(r'\W', '_', name.lower())}_hnsw"


class EmbeddingRepository:
    """Upkeep of the chunk_embeddings table: coverage per encoder, backfill, per-encoder indexes, drop.
    Ingestion writes and search reads stay on DocumentRepository."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def coverage(self) -> dict[str, int]:
        rows = await self._pool.fetch("SELECT encoder, count(*) AS n FROM chunk_embeddings GROUP BY encoder")
        return {r["encoder"]: r["n"] for r in rows}

    async def count_chunks(self) -> int:
        return await self._pool.fetchval("SELECT count(*) FROM chunks")

    async def count_missing(self, encoder: str) -> int:
        return await self._pool.fetchval(
            "SELECT count(*) FROM chunks c WHERE NOT EXISTS "
            "(SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id = c.id AND e.encoder = $1)",
            encoder,
        )

    async def missing(self, encoder: str, limit: int) -> list[asyncpg.Record]:
        return await self._pool.fetch(
            "SELECT c.id::text, c.content FROM chunks c WHERE NOT EXISTS "
            "(SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id = c.id AND e.encoder = $1) LIMIT $2",
            encoder,
            limit,
        )

    async def insert(self, encoder: str, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        await self._pool.executemany(
            "INSERT INTO chunk_embeddings (chunk_id, encoder, embedding) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            [(uuid.UUID(i), encoder, v) for i, v in zip(chunk_ids, vectors)],
        )

    async def ensure_index(self, encoder: str, dims: int) -> None:
        # CONCURRENTLY: ingestion keeps writing while the index builds
        await self._pool.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_index_name(encoder)} ON chunk_embeddings "
            f"USING hnsw ((embedding::vector({int(dims)})) vector_cosine_ops) WITH (m = 16, ef_construction = 64) "
            f"WHERE encoder = {encoder_literal(encoder)}"
        )

    async def drop(self, encoder: str) -> None:
        encoder_literal(encoder)  # rejects unsafe names before they reach the index name
        await self._pool.execute(f"DROP INDEX IF EXISTS {_index_name(encoder)}")
        await self._pool.execute("DELETE FROM chunk_embeddings WHERE encoder = $1", encoder)
