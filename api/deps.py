from typing import Annotated

from arq import ArqRedis
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from langgraph.graph.state import CompiledStateGraph

from db.repositories.documents import DocumentRepository
from db.repositories.query_log import QueryLogRepository
from settings import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key")


def get_repo(request: Request) -> DocumentRepository:
    return request.app.state.documents


def get_query_log(request: Request) -> QueryLogRepository:
    return request.app.state.query_log


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


def get_arq_pool(request: Request) -> ArqRedis:
    return request.app.state.arq_pool


DocumentsDep = Annotated[DocumentRepository, Depends(get_repo)]
QueryLogDep = Annotated[QueryLogRepository, Depends(get_query_log)]
GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
ArqPoolDep = Annotated[ArqRedis, Depends(get_arq_pool)]
ApiKey = Annotated[str, Security(_api_key_header)]


def verify_api_key(key: ApiKey) -> None:
    expected = get_settings().ingest_api_key.get_secret_value()
    if key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
