from typing import Literal

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[ConversationTurn] = []


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    sub_questions: list[str]
    clarifying_questions: list[str] = []
    run_id: str | None = None


class IngestUrlRequest(BaseModel):
    url: str
    mode: Literal["crawl", "single"] = "crawl"


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


class DocumentSummary(BaseModel):
    id: str
    source: str
    chunks: int
    ingested_at: str
