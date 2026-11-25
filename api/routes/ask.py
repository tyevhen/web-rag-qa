import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.sse import format_sse_event
from langgraph.graph.state import CompiledStateGraph

from agent.run import Run, run_graph
from api.deps import GraphDep, MessagesDep, ModelsDep, QueryLogDep, RequestTimeoutDep
from api.errors import record_failure
from api.limiter import limiter
from api.schemas import AskRequest, AskResponse, ModelsResponse
from db.repositories.messages import MessageRepository
from db.repositories.query_log import QueryLogRepository

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(request: Request, body: AskRequest, graph: GraphDep, query_log: QueryLogDep,
              messages: MessagesDep, timeout: RequestTimeoutDep, allowed: ModelsDep) -> AskResponse:
    run, history = await _start(request, body, messages, allowed)
    async for _ in run_graph(graph, run, history, timeout):  # raises RunTimeout / RunFailed, mapped in api/errors.py
        pass
    return await _finish(request, run, messages, query_log)


@router.post("/ask/stream", response_class=StreamingResponse)
@limiter.limit("10/minute")
async def ask_stream(request: Request, body: AskRequest, graph: GraphDep, query_log: QueryLogDep,
                     messages: MessagesDep, timeout: RequestTimeoutDep, allowed: ModelsDep) -> StreamingResponse:
    """Server-sent events: meta, then the answer as token events (or one status "held" when screen did not
    allow streaming), then done with the AskResponse, or error. A streamed answer is final: the output check
    after it only records a flag, and a flagged answer stays out of the conversation history."""
    run, history = await _start(request, body, messages, allowed)
    return StreamingResponse(_events(request, graph, run, history, timeout, messages, query_log),
                             media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


async def _events(request: Request, graph: CompiledStateGraph, run: Run, history: list[dict], timeout: float,
                  messages: MessageRepository, query_log: QueryLogRepository) -> AsyncIterator[bytes]:
    yield _sse("meta", {"run_id": run.run_id, "conversation_id": run.conversation_id, "model": run.model})
    try:
        async for kind, value in run_graph(graph, run, history, timeout, stream=True):
            yield _sse(kind, {"text" if kind == "token" else "status": value})
        yield _sse("done", (await _finish(request, run, messages, query_log)).model_dump())
    except Exception as exc:  # the 200 is already out, so a failure becomes an event; its row is written as usual
        _, detail = await record_failure(query_log, run, exc)
        yield _sse("error", {"detail": detail, "run_id": run.run_id})
    finally:
        with anyio.CancelScope(shield=True):  # a client that left mid-stream; a no-op once the row is written
            await run.record(query_log, "error", "client disconnected")


@router.get("/models", response_model=ModelsResponse)
async def list_models(allowed: ModelsDep) -> ModelsResponse:
    return ModelsResponse(default=next(iter(allowed)), models=list(allowed))


async def _start(request: Request, body: AskRequest, messages: MessageRepository,
                 allowed: dict[str, str]) -> tuple[Run, list[dict]]:
    model = body.model or next(iter(allowed))
    if model not in allowed:
        raise HTTPException(422, f"unknown model {model!r}; GET /models lists the choices")
    conversation_id = str(body.conversation_id or uuid.uuid4())  # an unknown id is just an empty conversation
    run = request.state.run = Run(run_id=request.state.run_id, question=body.question,  # error handlers record it
                                  model=model, conversation_id=conversation_id)
    return run, await messages.last_turns(conversation_id, model)


async def _finish(request: Request, run: Run, messages: MessageRepository, query_log: QueryLogRepository) -> AskResponse:
    if run.keeps_history:
        await messages.append(str(run.conversation_id), str(run.model), run.question, run.answer)
    await run.record(query_log)
    return AskResponse.from_run(run, str(request.base_url))


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return format_sse_event(event=event, data_str=json.dumps(data))
