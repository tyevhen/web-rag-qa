from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from settings import get_settings


@dataclass(frozen=True)
class Encoder:
    name: str  # sentence-transformers id; also the label stored with each vector in chunk_embeddings
    dims: int
    max_tokens: int  # the window, [CLS] and [SEP] included; longer text is cut off
    query_prompt: str | None = None  # put before short queries that search longer passages; passages get none


ENCODERS = {e.name: e for e in (
    Encoder("BAAI/bge-small-en-v1.5", 384, 512, "Represent this sentence for searching relevant passages: "),
    Encoder("sentence-transformers/all-MiniLM-L6-v2", 384, 256),
)}


def get_encoder(name: str | None = None) -> Encoder:
    """The named encoder, or the one search uses (Settings.embedder)."""
    name = name or get_settings().embedder
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; known: {', '.join(ENCODERS)}")
    return ENCODERS[name]


def write_encoders() -> list[Encoder]:
    """Every encoder ingestion writes: the search one plus any shadow ones being prepared or compared."""
    names = dict.fromkeys([get_settings().embedder, *get_settings().shadow_embedders])
    return [get_encoder(n) for n in names]


@lru_cache
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def _encode(texts: list[str], encoder: Encoder, prompt: str | None) -> list[list[float]]:
    vecs = _model(encoder.name).encode(texts, prompt=prompt, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def count_tokens(text: str, encoder: Encoder | None = None) -> int:
    return len(_model((encoder or get_encoder()).name).tokenizer(text)["input_ids"])


def embed_queries(texts: list[str], encoder: Encoder | None = None) -> list[list[float]]:
    encoder = encoder or get_encoder()
    return _encode(texts, encoder, encoder.query_prompt)


def embed_query(text: str, encoder: Encoder | None = None) -> list[float]:
    return embed_queries([text], encoder)[0]


def embed_documents(texts: list[str], encoder: Encoder | None = None) -> list[list[float]]:
    return _encode(texts, encoder or get_encoder(), None)
