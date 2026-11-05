import logging

from arq.jobs import Job, JobStatus
from fastapi import APIRouter, HTTPException

from api.deps import ArqPoolDep
from api.schemas import IngestJobAccepted, IngestJobResult, IngestJobStatus, IngestUrlRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest")


@router.post("/url", status_code=202, response_model=IngestJobAccepted)
async def ingest_from_url(body: IngestUrlRequest, arq_pool: ArqPoolDep) -> IngestJobAccepted:
    job = await arq_pool.enqueue_job("ingest_url_task", body.url, body.mode)
    if job is None:
        raise HTTPException(status_code=409, detail="A job for this URL is already queued")
    logger.info("Enqueued ingestion job %s for %s", job.job_id, body.url)
    return IngestJobAccepted(job_id=job.job_id)


@router.get("/jobs/{job_id}", response_model=IngestJobStatus)
async def get_ingest_job(job_id: str, arq_pool: ArqPoolDep) -> IngestJobStatus:
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
