import hashlib
import logging

from langdetect import DetectorFactory, LangDetectException, detect

from ingestion.scraper import RawChunk

DetectorFactory.seed = 0

log = logging.getLogger(__name__)
_MIN_WORDS = 20
_MIN_ALPHA_RATIO = 0.5


def _lang(text: str) -> str | None:
    try:
        return detect(text)
    except LangDetectException:
        return None


def evaluate_chunks(chunks: list[RawChunk]) -> list[RawChunk]:
    seen: set[str] = set()
    result: list[RawChunk] = []
    for chunk in chunks:
        if len(chunk.content.split()) < _MIN_WORDS:
            continue
        alpha = sum(1 for c in chunk.content if c.isalpha())
        if alpha / max(len(chunk.content), 1) < _MIN_ALPHA_RATIO:
            continue
        fp = hashlib.md5(chunk.content.encode()).hexdigest()
        if fp in seen:
            continue
        seen.add(fp)
        if _lang(chunk.content) != "en":
            log.debug("Dropping non-English chunk from %s", chunk.metadata.get("url", ""))
            continue
        result.append(chunk)
    log.info("Evaluated %d → %d chunks (English only)", len(chunks), len(result))
    return result
