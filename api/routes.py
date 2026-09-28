import logging
import time
import uuid
from urllib.parse import urlparse

import structlog

from arq import ArqRedis
from arq.jobs import Job, JobStatus
from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from api.deps import get_arq_pool, get_graph, get_repo
from api.limiter import limiter
from api.schemas import (
    AskRequest,
    AskResponse,
    DocumentSummary,
    IngestJobAccepted,
    IngestJobResult,
    IngestJobStatus,
    IngestUrlRequest,
)
from db.repository import DocumentRepository, QueryLogEntry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(
    request: Request,
    body: AskRequest,
    graph=Depends(get_graph),
    repo: DocumentRepository = Depends(get_repo),
) -> AskResponse:
    run_id = uuid.uuid4()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=str(run_id))
    config: RunnableConfig = {
        "run_id": run_id,
        "run_name": "web-rag-qa",
        "metadata": {"question": body.question[:200]},
        "tags": ["web-rag-qa"],
    }
    initial: AgentState = {
        "question": body.question,
        "history": [{"role": t.role, "content": t.content} for t in body.history],
        "sub_questions": [],
        "retrieved_chunks": [],
        "graded_chunks": [],
        "answer": "",
        "citations": [],
        "clarifying_questions": [],
        "retry_count": 0,
    }
    t0 = time.monotonic()
    try:
        result = await graph.ainvoke(initial, config=config)
    except Exception as exc:
        logger.exception("Graph execution failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    latency_ms = int((time.monotonic() - t0) * 1000)
    route = "clarify" if result["clarifying_questions"] else "reason"
    domain = urlparse(result["citations"][0]).netloc if result["citations"] else None

    try:
        await repo.log_query(QueryLogEntry(
            run_id=str(run_id),
            question=body.question,
            answer=result["answer"],
            domain=domain,
            latency_ms=latency_ms,
            route=route,
            sub_question_count=len(result["sub_questions"]),
            retrieved_chunk_count=len(result["retrieved_chunks"]),
            graded_chunk_count=len(result["graded_chunks"]),
            retry_count=result["retry_count"],
            citations=result["citations"],
        ))
    except Exception:
        logger.exception("Failed to write query_log for run_id=%s", run_id)

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        sub_questions=result["sub_questions"],
        clarifying_questions=result["clarifying_questions"],
        run_id=str(run_id),
    )


@router.post("/ingest/url", status_code=202, response_model=IngestJobAccepted)
async def ingest_from_url(
    body: IngestUrlRequest,
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> IngestJobAccepted:
    job = await arq_pool.enqueue_job("ingest_url_task", body.url, body.mode)
    if job is None:
        raise HTTPException(status_code=409, detail="A job for this URL is already queued")
    logger.info("Enqueued ingestion job %s for %s", job.job_id, body.url)
    return IngestJobAccepted(job_id=job.job_id)


@router.get("/ingest/jobs/{job_id}", response_model=IngestJobStatus)
async def get_ingest_job(
    job_id: str,
    arq_pool: ArqRedis = Depends(get_arq_pool),
) -> IngestJobStatus:
    job = Job(job_id=job_id, redis=arq_pool)
    status = await job.status()
    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    result = None
    if status == JobStatus.complete:
        info = await job.result_info()
        if info and not info.success:
            raise HTTPException(status_code=500, detail="Ingestion job failed")
        if info and info.result:
            result = IngestJobResult(**info.result)

    return IngestJobStatus(job_id=job_id, status=status.value, result=result)


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    repo: DocumentRepository = Depends(get_repo),
) -> list[DocumentSummary]:
    rows = await repo.list_documents()
    return [
        DocumentSummary(
            id=row["id"],
            source=row["filename"],
            chunks=row["chunks"],
            ingested_at=row["uploaded_at"].isoformat(),
        )
        for row in rows
    ]


@router.get("/topics", response_model=list[str])
async def list_topics(
    repo: DocumentRepository = Depends(get_repo),
) -> list[str]:
    return await repo.list_sections()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
