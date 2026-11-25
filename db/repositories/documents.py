import uuid

import asyncpg
from asyncpg import Pool

from db.repositories.embeddings import encoder_literal


def _embedding_rows(ids: list[uuid.UUID], embeddings: dict[str, list[list[float]]]) -> list[tuple]:
    """One (chunk_id, encoder, vector) row per chunk for each encoder."""
    rows = []
    for encoder, vectors in embeddings.items():
        for chunk_id, vector in zip(ids, vectors, strict=True):
            rows.append((chunk_id, encoder, vector))
    return rows


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
                _embedding_rows(ids, embeddings),
            )

    async def similarity_search(self, embedding: list[float], k: int, encoder: str, lang: str) -> list[asyncpg.Record]:
        # Encoder and size inlined so the query matches that encoder's partial index (db/repositories/embeddings.py)
        # shortcut: with no stats on metadata->>'lang' the planner picks an exact scan (12 ms at 5.5k chunks)
        # over HNSW; add an index on (metadata->>'lang') or extended stats if latency grows with the corpus
        vec =f"e.embedding::vector({len(embedding)})"
        return await self._pool.fetch(
            f"""
            SELECT c.id::text, c.content, c.metadata, 1 - ({vec} <=> $1) AS score
            FROM chunk_embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            WHERE e.encoder = {encoder_literal(encoder)} AND c.metadata->>'lang' = $3
            ORDER BY {vec} <=> $1
            LIMIT $2
            """,
            embedding,
            k,
            lang,
        )

    async def fulltext_search(self, query: str, k: int, lang: str) -> list[asyncpg.Record]:
        """BM25 (k1 1.2, b 0.75) over chunks matching any question term, terms made as for the chunks
        (keyword_vector, migrations/011_keyword_bm25.sql). Terms with symbols (URLs, hyphenated) are left out
        of the query so it is a valid tsquery; their word parts are terms of their own."""
        # shortcut: chunk count and mean length are recomputed per query (a scan of the language's chunks);
        # keep them in a table updated on ingest if keyword search gets slow as the corpus grows
        return await self._pool.fetch(
            r"""
            WITH q AS (
                SELECT ARRAY(SELECT t FROM unnest(tsvector_to_array(keyword_vector($3, $1))) t WHERE t ~ '^\w+$') AS terms
            ), m AS (
                SELECT c.id, c.fts_len, t.lexeme, cardinality(t.positions) AS tf
                FROM q, chunks c, unnest(c.fts) t
                WHERE c.fts @@ array_to_string(q.terms, ' | ')::tsquery
                  AND c.metadata->>'lang' = $3 AND t.lexeme = ANY(q.terms)
            ), s AS (
                SELECT count(*) AS n, avg(fts_len) AS avgdl FROM chunks WHERE metadata->>'lang' = $3
            ), d AS (
                SELECT lexeme, count(*) AS df FROM m GROUP BY lexeme
            ), r AS (
                SELECT m.id, sum(ln(1 + (s.n - d.df + 0.5) / (d.df + 0.5))
                                 * m.tf * 2.2 / (m.tf + 1.2 * (0.25 + 0.75 * m.fts_len / s.avgdl))) AS score
                FROM m JOIN d USING (lexeme) CROSS JOIN s
                GROUP BY m.id
                ORDER BY score DESC, m.id
                LIMIT $2
            )
            SELECT c.id::text, c.content, c.metadata
            FROM r JOIN chunks c USING (id)
            ORDER BY r.score DESC, c.id
            """,
            query,
            k,
            lang,
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
