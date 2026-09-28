from arq import ArqRedis
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from langgraph.graph.state import CompiledStateGraph

from agent.config import TopicConfig
from db.repository import DocumentRepository
from settings import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key")


def get_repo(request: Request) -> DocumentRepository:
    return request.app.state.repo


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


def get_topic_config(request: Request) -> TopicConfig:
    return request.app.state.topic_config


def get_arq_pool(request: Request) -> ArqRedis:
    return request.app.state.arq_pool


def verify_api_key(key: str = Security(_api_key_header)) -> None:
    expected = get_settings().ingest_api_key.get_secret_value()
    if key != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
