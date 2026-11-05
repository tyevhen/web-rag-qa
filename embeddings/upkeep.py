"""Encoder upkeep for the vector store. Switching encoders without downtime:

1. Add the encoder to agent/embed.py ENCODERS and to SHADOW_EMBEDDERS; deploy (ingestion now writes both).
2. uv run python -m embeddings backfill NAME   # vectors for the chunks ingested before, then its index
3. Set EMBEDDER=NAME and restart (startup refuses an encoder with missing vectors).
4. Remove the old one from SHADOW_EMBEDDERS, then: uv run python -m embeddings drop OLD_NAME

    uv run python -m embeddings status
"""
import asyncio
import logging

from agent.embed import embed_documents, get_encoder, write_encoders
from db.repositories.embeddings import EmbeddingRepository

log = logging.getLogger(__name__)
_BATCH_SIZE = 64


async def check_ready(vectors: EmbeddingRepository) -> None:
    """Raises if search's encoder lacks a vector for any chunk; also validates every configured encoder name."""
    write_encoders()
    missing = await vectors.count_missing(get_encoder().name)
    if missing:
        raise RuntimeError(f"{missing} chunks have no {get_encoder().name} vector: backfill it before switching")


async def backfill(vectors: EmbeddingRepository, name: str) -> int:
    encoder, done = get_encoder(name), 0
    while rows := await vectors.missing(encoder.name, _BATCH_SIZE):
        embedded = await asyncio.to_thread(embed_documents, [r["content"] for r in rows], encoder)
        await vectors.insert(encoder.name, [r["id"] for r in rows], embedded)
        done += len(rows)
        log.info("%s: %d vectors written", encoder.name, done)
    await vectors.ensure_index(encoder.name, encoder.dims)
    return done


async def drop(vectors: EmbeddingRepository, name: str) -> None:
    if name in {e.name for e in write_encoders()}:
        raise ValueError(f"{name} is still in use (EMBEDDER or SHADOW_EMBEDDERS); remove it there first")
    await vectors.drop(name)
