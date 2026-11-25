from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi.errors import RateLimitExceeded

from agent.config import get_agent_config
from agent.graph import build_graph
from agent.nodes._llm import get_llm
from agent.run import RunError
from api.errors import http_error_handler, rate_limit_handler, run_error_handler, validation_error_handler
from api.limiter import limiter
from api.logging_config import configure_logging
from api.middleware import RequestContextMiddleware
from api.routes import router
from db.client import create_pool
from db.repositories.documents import DocumentRepository
from db.repositories.embeddings import EmbeddingRepository
from db.repositories.messages import MessageRepository
from db.repositories.query_log import QueryLogRepository
from embeddings.upkeep import check_ready
from settings import get_settings

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # validate all env vars before accepting traffic

    settings.export_langsmith_env()

    get_llm()  # fail fast on a bad LLM config, not on the first /ask
    config = get_agent_config()
    pool = await create_pool()
    await check_ready(EmbeddingRepository(pool))  # refuse to serve search over a half-backfilled encoder
    documents = DocumentRepository(pool)
    app.state.documents = documents
    app.state.query_log = QueryLogRepository(pool)
    app.state.messages = MessageRepository(pool)
    app.state.graph = build_graph(config, documents)
    yield
    if arq_pool := getattr(app.state, "arq_pool", None):  # connected on the first ingest request (api/deps.py)
        await arq_pool.close()
    await pool.close()


app = FastAPI(title="Web RAG Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RunError, run_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(StarletteHTTPException, http_error_handler)  # type: ignore[arg-type]
app.add_middleware(RequestContextMiddleware)  # added before CORS, so CORS wraps it and 500s keep CORS headers

_settings = get_settings()
origins = _settings.allowed_origins.split(",") if _settings.allowed_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
