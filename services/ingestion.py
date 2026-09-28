import asyncio
import logging
from urllib.parse import urlparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.embed import embed_documents as _embed_documents
from agent.nodes._llm import get_groq_llm
from db.repository import DocumentRepository
from ingestion.chunker import chunk_raw
from ingestion.evaluator import evaluate_chunks
from ingestion.scraper import RawChunk, scrape_single, scrape_site

log = logging.getLogger(__name__)
_BATCH_SIZE = 20

_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise summarizer."),
    ("human", "Summarize what this website is about in 2-3 sentences:\n\n{content}"),
])


async def _embed_and_store(repo: DocumentRepository, doc_id: str, chunks: list[RawChunk]) -> None:
    for start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[start : start + _BATCH_SIZE]
        contents = [c.content for c in batch]
        metadatas = [{"document_id": doc_id, "chunk_index": start + i, **c.metadata} for i, c in enumerate(batch)]
        embeddings = await asyncio.to_thread(_embed_documents, contents)
        await repo.insert_chunks(doc_id, contents, metadatas, embeddings)
        log.info("Embedded batch %d–%d", start, start + len(batch) - 1)


async def ingest_url(url: str, repo: DocumentRepository, crawler, mode: str = "crawl") -> tuple[str, int]:
    raw = await (scrape_single(url, crawler) if mode == "single" else scrape_site(url, crawler))
    if not raw:
        raise RuntimeError(f"No content scraped from {url!r}")

    chunks = chunk_raw(raw, "markdown_headers", ["url", "source", "section"])
    chunks = evaluate_chunks(chunks)
    if not chunks:
        raise RuntimeError("No usable chunks after evaluation")
    log.info("Scraped %d pages → %d chunks (after evaluation)", len(raw), len(chunks))

    domain = urlparse(url).netloc
    doc_id = await repo.create_document(url, domain)
    await _embed_and_store(repo, doc_id, chunks)

    combined = "\n\n".join(r.content[:500] for r in raw[:10])
    chain = _summary_prompt | get_groq_llm() | StrOutputParser()
    summary = (await chain.ainvoke({"content": combined})).strip()
    log.info("Done. %d chunks ingested from %s", len(chunks), url)
    return summary, len(raw)
