"""Encoder upkeep for the vector store. Switching encoders without downtime:

1. Add the encoder to agent/embed.py ENCODERS and to SHADOW_EMBEDDERS; deploy (ingestion now writes both).
2. uv run python -m services.encoders backfill NAME   # vectors for the chunks ingested before, then its index
3. Set EMBEDDER=NAME and restart (startup refuses an encoder with missing vectors).
4. Remove the old one from SHADOW_EMBEDDERS, then: uv run python -m services.encoders drop OLD_NAME

    uv run python -m services.encoders status
"""
import argparse
import asyncio
import logging

from agent.embed import embed_documents, get_encoder, write_encoders
from db.client import create_pool
from db.embedding_repository import EmbeddingRepository

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


async def _main(command: str, name: str | None) -> None:
    pool = await create_pool()
    try:
        vectors = EmbeddingRepository(pool)
        if command == "backfill" and name:
            log.info("%s: backfilled %d chunks", name, await backfill(vectors, name))
        elif command == "drop" and name:
            await drop(vectors, name)
            log.info("%s: vectors and index dropped", name)
        total, coverage = await vectors.count_chunks(), await vectors.coverage()
        for encoder, n in coverage.items():
            role = "search" if encoder == get_encoder().name else "stored"
            log.info("%-45s %6d / %d chunks  (%s)", encoder, n, total, role)
    finally:
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status", "backfill", "drop"])
    parser.add_argument("name", nargs="?", help="encoder name from agent/embed.py ENCODERS")
    args = parser.parse_args()
    if args.command != "status" and not args.name:
        parser.error(f"{args.command} needs an encoder name")
    asyncio.run(_main(args.command, args.name))
