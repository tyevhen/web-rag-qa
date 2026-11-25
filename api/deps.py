from typing import Annotated

import arq
from arq import ArqRedis
from arq.connections import RedisSettings
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from langgraph.graph.state import CompiledStateGraph

from agent.nodes._llm import models
from db.repositories.documents import DocumentRepository
from db.repositories.messages import MessageRepository
from db.repositories.query_log import QueryLogRepository
from settings import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key")


def get_repo(request: Request) -> DocumentRepository:
    return request.app.state.documents


def get_query_log(request: Request) -> QueryLogRepository:
    return request.app.state.query_log


def get_messages(request: Request) -> MessageRepository:
    return request.app.state.messages


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


async def get_arq_pool(request: Request) -> ArqRedis:
    """Connected on first use, so the API serves /ask without Redis; only ingestion needs it."""
    state = request.app.state
    if getattr(state, "arq_pool", None) is None:  # shortcut: two first requests at once may open two pools; lock it if that matters
        state.arq_pool = await arq.create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return state.arq_pool


def get_request_timeout() -> float:
    return get_settings().request_timeout


DocumentsDep = Annotated[DocumentRepository, Depends(get_repo)]
QueryLogDep = Annotated[QueryLogRepository, Depends(get_query_log)]
MessagesDep = Annotated[MessageRepository, Depends(get_messages)]
GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
ArqPoolDep = Annotated[ArqRedis, Depends(get_arq_pool)]
RequestTimeoutDep = Annotated[float, Depends(get_request_timeout)]
ModelsDep = Annotated[dict[str, str], Depends(models)]  # model -> base URL, the default first
ApiKey = Annotated[str, Security(_api_key_header)]


def verify_api_key(key: ApiKey) -> None:
    expected = get_settings().ingest_api_key.get_secret_value()
    if key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
