from typing import Annotated, Any, Literal
from uuid import UUID

from arq.jobs import JobResult, JobStatus
from pydantic import BaseModel, ConfigDict, Field

from agent.nodes.formatter import text_fragment
from agent.run import Run


class HealthResponse(BaseModel):
    status: str = "ok"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a client still sending `history` gets a 422

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID | None = None  # omit to start a new conversation; the response returns its id
    model: str | None = None  # one of GET /models; omit for the app's model. Each model has its own history.


class Citation(BaseModel):
    n: int  # the [n] marker in the answer
    chunk_id: str
    section: str
    source_url: str  # the page, with a text fragment that scrolls to the passage where browsers support it
    url: str  # GET /chunks/{chunk_id}: the passage as stored


class AskResponse(BaseModel):
    answer: str  # plain [n] markers; citations say where each one points
    citations: list[Citation]
    sub_questions: list[str]
    clarifying_questions: list[str] = []
    run_id: str
    conversation_id: str
    model: str

    @classmethod
    def from_run(cls, run: Run, base_url: str) -> "AskResponse":
        """`base_url` is the API's own, ending in "/", for each citation's absolute `url`."""
        s = run.state
        citations = [Citation(**c, url=f"{base_url}chunks/{c['chunk_id']}") for c in s["citations"]]
        return cls(answer=run.answer, citations=citations, sub_questions=s["sub_questions"],
                   clarifying_questions=s["clarifying_questions"], run_id=run.run_id,
                   conversation_id=str(run.conversation_id), model=str(run.model))


class ChunkResponse(BaseModel):
    id: str
    content: str
    source: str  # the site's netloc
    section: str
    source_url: str  # with a text fragment, like a citation's

    @classmethod
    def from_row(cls, row: Any) -> "ChunkResponse":  # row is an asyncpg.Record
        meta = row["metadata"]
        return cls(id=row["id"], content=row["content"], source=meta.get("source", ""),
                   section=meta.get("section", ""),
                   source_url=meta.get("url", "") + text_fragment(row["content"]))


class ModelsResponse(BaseModel):
    default: str
    models: list[str]


class IngestUrlRequest(BaseModel):
    url: str
    mode: Literal["crawl", "single", "pdf"] = "crawl"  # pdf: the local copy in sources/ (ingestion/pdf.py)
    langs: list[Annotated[str, Field(pattern=r"^[a-z]{2}$")]] = Field(default=["en"], min_length=1)  # pages kept
    exclude_paths: list[Annotated[str, Field(pattern=r"^/")]] = []  # URL path prefixes not crawled, e.g. "/en"


class IngestJobAccepted(BaseModel):
    job_id: str
    status: str = "queued"


class IngestJobResult(BaseModel):
    url: str
    pages_scraped: int
    summary: str


class IngestJobStatus(BaseModel):
    job_id: str
    status: str
    result: IngestJobResult | None = None
    error: str | None = None  # set when status is "failed"

    @classmethod
    def from_job(cls, job_id: str, status: JobStatus, info: JobResult | None) -> "IngestJobStatus":
        """A failed job is status "failed", not an HTTP error; only the exception type is shown (the route is public)."""
        if status != JobStatus.complete or info is None:
            return cls(job_id=job_id, status=status.value)
        if not info.success:
            return cls(job_id=job_id, status="failed", error=type(info.result).__name__)
        return cls(job_id=job_id, status=status.value, result=IngestJobResult(**info.result) if info.result else None)


class DocumentSummary(BaseModel):
    id: str
    source: str
    chunks: int
    ingested_at: str

    @classmethod
    def from_row(cls, row: Any) -> "DocumentSummary":  # row is an asyncpg.Record
        return cls(id=row["id"], source=row["filename"], chunks=row["chunks"], ingested_at=row["uploaded_at"].isoformat())
