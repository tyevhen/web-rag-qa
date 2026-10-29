from typing import cast

import pytest

from agent.embed import ENCODERS, get_encoder
from db.embedding_repository import EmbeddingRepository, encoder_literal
from db.repository import DocumentRepository
from ingestion.scraper import RawChunk
from services import encoders, ingestion

_BGE, _MINI = ENCODERS["BAAI/bge-small-en-v1.5"], ENCODERS["sentence-transformers/all-MiniLM-L6-v2"]


class FakeRepo:
    def __init__(self) -> None:
        self.inserted: list[dict[str, list[list[float]]]] = []

    async def insert_chunks(self, doc_id: str, contents: list[str], metadatas: list[dict],
                            embeddings: dict[str, list[list[float]]]) -> None:
        self.inserted.append(embeddings)


async def test_ingestion_writes_a_vector_per_chunk_for_every_encoder(monkeypatch):
    monkeypatch.setattr(ingestion, "write_encoders", lambda: [_BGE, _MINI])
    monkeypatch.setattr(ingestion, "_embed_documents", lambda texts, e: [[float(e.max_tokens)] for _ in texts])
    repo = FakeRepo()
    await ingestion._embed_and_store(cast(DocumentRepository, repo), "doc", [RawChunk("a", {}), RawChunk("b", {})])

    assert repo.inserted == [{_BGE.name: [[512.0], [512.0]], _MINI.name: [[256.0], [256.0]]}]


def test_unknown_encoder_is_refused():
    with pytest.raises(ValueError, match="Unknown encoder"):
        get_encoder("no/such-model")


def test_encoder_name_cannot_inject_sql():
    assert encoder_literal(_BGE.name) == "'BAAI/bge-small-en-v1.5'"
    with pytest.raises(ValueError, match="Unsafe"):
        encoder_literal("x' OR '1'='1")


async def test_drop_refuses_an_encoder_still_in_use(monkeypatch):
    monkeypatch.setattr(encoders, "write_encoders", lambda: [_BGE])
    with pytest.raises(ValueError, match="still in use"):
        await encoders.drop(cast(EmbeddingRepository, object()), _BGE.name)
