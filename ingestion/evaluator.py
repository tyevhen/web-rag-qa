import hashlib
import logging

from agent.lang import detect_lang
from ingestion.scraper import RawChunk

log = logging.getLogger(__name__)
_MIN_WORDS = 20
_MIN_ALPHA_RATIO = 0.5


def filter_pages(pages: list[RawChunk], langs: list[str]) -> list[RawChunk]:
    """Keep whole pages in `langs` and tag them with metadata["lang"]; a page is a far surer sample than a chunk."""
    kept: list[RawChunk] = []
    for page in pages:
        lang = detect_lang(page.content)
        if lang not in langs:
            log.debug("Dropping %s page %s", lang, page.metadata.get("url", ""))
            continue
        kept.append(RawChunk(content=page.content, metadata={**page.metadata, "lang": lang}))
    log.info("filter_pages: kept %d, rejected %d (langs=%s)", len(kept), len(pages) - len(kept), langs)
    return kept


def evaluate_chunks(chunks: list[RawChunk], langs: list[str]) -> list[RawChunk]:
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
        lang = detect_lang(chunk.content)
        # Its page already passed the filter; detection on short Estonian text is unreliable, so give it the benefit
        if lang not in langs and not (lang is None and "et" in langs):
            log.debug("Dropping %s chunk from %s", lang, chunk.metadata.get("url", ""))
            continue
        result.append(chunk)
    log.info("Evaluated %d → %d chunks (langs=%s)", len(chunks), len(result), langs)
    return result
