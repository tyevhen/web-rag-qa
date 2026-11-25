import logging
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException

from agent.run import Run, RunError, RunTimeout
from api.deps import get_query_log
from db.repositories.query_log import QueryLogRepository

logger = logging.getLogger(__name__)


def describe_errors(errors: Sequence[Any]) -> str:
    """A short reason from pydantic errors, e.g. 'question: 41,203 chars > 2,000'."""
    parts = []
    for e in errors:
        loc = ".".join(str(x) for x in e.get("loc", ()) if x != "body") or "body"
        limit = e.get("ctx", {}).get("max_length")
        value = e.get("input")
        if limit is not None and isinstance(value, str | list):
            unit = "chars" if isinstance(value, str) else "items"
            parts.append(f"{loc}: {len(value):,} {unit} > {limit:,}")
        else:
            parts.append(f"{loc}: {e.get('msg', 'invalid')}")
    return "; ".join(parts)


def _question_of(body: Any) -> str:
    """The question text, from a parsed body or, when the JSON was invalid, the raw text."""
    question = body.get("question") if isinstance(body, dict) else body
    return question if isinstance(question, str) else ""


def _error_body(status: int, detail: str, run_id: str | None, headers: Mapping[str, str] | None = None) -> JSONResponse:
    """The one error body: {"detail", "run_id"}; only the 422 keeps FastAPI's list of validation errors."""
    return JSONResponse({"detail": detail, "run_id": run_id}, status_code=status, headers=headers)


async def _reject(request: Request, question: str, reason: str) -> None:
    """A 'rejected' query_log row for an /ask request that never ran."""
    if getattr(request.scope.get("route"), "name", None) == "ask":  # the handler's name, not its path
        await Run(run_id=request.state.run_id, question=question).record(get_query_log(request), "rejected", reason)


async def record_failure(query_log: QueryLogRepository, run: Run, exc: Exception) -> tuple[int, str]:
    """A timed-out or failed run: one log line and one query_log row with the reason; the status and client message."""
    if isinstance(exc, RunTimeout):
        logger.warning("Graph run %s: %s", run.run_id, exc.reason)
        await run.record(query_log, "timeout", exc.reason)
        return 504, "This took too long, please try again"
    reason = exc.reason if isinstance(exc, RunError) else f"{type(exc).__name__}: {str(exc)[:300]}"
    logger.error("Graph run %s failed: %s", run.run_id, reason, exc_info=exc)
    await run.record(query_log, "error", reason)
    return 500, "Internal server error"


async def run_error_handler(request: Request, exc: RunError) -> Response:
    status, detail = await record_failure(get_query_log(request), exc.run, exc)
    return _error_body(status, detail, exc.run.run_id)


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    """Catch-all: anything no other handler maps (DB, Redis, bugs) is logged once and becomes a 500."""
    run_id = getattr(request.state, "run_id", None)
    logger.error("Unhandled error in %s %s (run_id=%s)", request.method, request.url.path, run_id, exc_info=exc)
    if run := getattr(request.state, "run", None):
        await run.record(get_query_log(request), "error", f"{type(exc).__name__}: {str(exc)[:300]}")
    return _error_body(500, "Internal server error", run_id)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> Response:
    """Log a 'rejected' query_log row for an invalid /ask request, then return FastAPI's usual 422."""
    await _reject(request, _question_of(exc.body), describe_errors(exc.errors()))
    return await request_validation_exception_handler(request, exc)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Log a 'rejected' row for a rate-limited /ask request, then a 429 with slowapi's rate-limit headers."""
    await _reject(request, "", f"rate limit: {exc.detail}")  # the body was parsed by the route, not kept here
    response = _error_body(429, f"Rate limit exceeded: {exc.detail}", request.state.run_id)
    return request.app.state.limiter._inject_headers(response, request.state.view_rate_limit)


async def http_error_handler(request: Request, exc: HTTPException) -> Response:
    """Expected 4xx raised in routes (404, 409) and unknown paths, in the same body as every other error."""
    return _error_body(exc.status_code, str(exc.detail), getattr(request.state, "run_id", None), exc.headers)

