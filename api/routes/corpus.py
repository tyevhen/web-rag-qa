import html
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from api.deps import DocumentsDep
from api.schemas import ChunkResponse, DocumentSummary

router = APIRouter()

# Scraped text is shown as plain text: everything from the page is escaped, nothing is rendered as markup
_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;color:#222;background:#fff}}
pre{{white-space:pre-wrap;font:inherit;background:#f6f6f4;padding:1rem;border-radius:6px}}
@media (prefers-color-scheme:dark){{body{{color:#ddd;background:#161616}}pre{{background:#232323}}}}</style></head>
<body><h1>{title}</h1><p>{source}</p><pre>{content}</pre></body></html>"""


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(repo: DocumentsDep) -> list[DocumentSummary]:
    return [DocumentSummary.from_row(row) for row in await repo.list_documents()]


@router.get("/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(chunk_id: UUID, request: Request, repo: DocumentsDep) -> ChunkResponse | HTMLResponse:
    """One stored passage, the target of a citation's `url`: JSON, or a readable page when a browser asks."""
    rows = await repo.get_chunks_by_ids([str(chunk_id)])
    if not rows:
        raise HTTPException(404, "no such chunk: re-ingesting a site replaces its chunks")
    chunk = ChunkResponse.from_row(rows[0])
    if "text/html" not in request.headers.get("accept", ""):
        return chunk
    return HTMLResponse(_PAGE.format(title=html.escape(chunk.section or chunk.source),
                                     source=_source_link(chunk.source_url), content=html.escape(chunk.content)))


def _source_link(url: str) -> str:
    if not url.startswith(("https://", "http://")):  # only web links become clickable
        return html.escape(url)
    return f'<a href="{html.escape(url)}">Open the source page</a>'
