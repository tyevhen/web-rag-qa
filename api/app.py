import os
from contextlib import asynccontextmanager

import arq
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from agent.config import get_agent_config
from agent.graph import build_graph
from agent.nodes._llm import get_groq_llm
from api.limiter import limiter
from api.logging_config import configure_logging
from api.routes import router
from db.client import create_pool
from db.embedding_repository import EmbeddingRepository
from db.query_log_repository import QueryLogRepository
from db.repository import DocumentRepository
from services.encoders import check_ready
from settings import get_settings

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # validate all env vars before accepting traffic

    if settings.langchain_tracing_v2 and settings.langchain_api_key is not None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()
        if settings.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    get_groq_llm()  # fail fast on a bad LLM config (e.g. HF URL without HF_TOKEN), not on the first /ask
    config = get_agent_config()
    pool = await create_pool()
    await check_ready(EmbeddingRepository(pool))  # refuse to serve search over a half-backfilled encoder
    documents = DocumentRepository(pool)
    app.state.documents = documents
    app.state.query_log = QueryLogRepository(pool)
    app.state.graph = build_graph(config, documents)
    app.state.topic_config = config
    arq_pool = await arq.create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.arq_pool = arq_pool
    yield
    await arq_pool.close()
    await pool.close()


app = FastAPI(title="Web RAG Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

_settings = get_settings()
origins = _settings.allowed_origins.split(",") if _settings.allowed_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
