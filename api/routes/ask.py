import logging
import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request
from langchain_core.runnables import RunnableConfig

from agent.graph import log_entry
from agent.state import initial_state
from api.deps import GraphDep, QueryLogDep
from api.limiter import limiter
from api.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(request: Request, body: AskRequest, graph: GraphDep, query_log: QueryLogDep) -> AskResponse:
    run_id = uuid.uuid4()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=str(run_id))
    config: RunnableConfig = {
        "run_id": run_id,
        "run_name": "web-rag-qa",
        "metadata": {"question": body.question[:200]},
        "tags": ["web-rag-qa"],
    }
    initial = initial_state(body.question, [{"role": t.role, "content": t.content} for t in body.history])
    t0 = time.monotonic()
    try:
        result = await graph.ainvoke(initial, config=config)
    except Exception as exc:
        logger.exception("Graph execution failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    latency_ms = int((time.monotonic() - t0) * 1000)
    try:
        await query_log.log_query(log_entry(str(run_id), body.question, result, latency_ms))
    except Exception:
        logger.exception("Failed to write query_log for run_id=%s", run_id)

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        sub_questions=result["sub_questions"],
        clarifying_questions=result["clarifying_questions"],
        run_id=str(run_id),
    )
