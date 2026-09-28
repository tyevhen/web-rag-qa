# web-rag-qa

Agentic RAG question-answering service. Point it at a website — it crawls the pages, embeds the content into Postgres/pgvector, and answers natural-language questions over it with source citations.

The deployment in this repo is configured as an **emergency preparedness assistant** over two public sources: Estonian disaster preparedness guidance ([olevalmis.ee](https://www.olevalmis.ee)) and Ukrainian tactical combat casualty care ([tccc.org.ua/en](https://tccc.org.ua/en)). The domain is set entirely by the `SYSTEM_PROMPT` environment variable — the pipeline itself is topic-agnostic.

## How it works

### Ingestion

`POST /ingest/url` enqueues a background job (arq + Redis). The worker:

1. **Crawls** the site's sitemap with crawl4ai (Playwright-rendered, JS-capable), or a single page in `single` mode.
2. **Chunks** each page on Markdown headings, keeping `url`, `source` and `section` metadata.
3. **Filters** chunks: under 20 words, under 50% alphabetic characters, exact duplicates (MD5) and non-English text (langdetect) are dropped.
4. **Embeds** chunks with `all-MiniLM-L6-v2` (384-dim, L2-normalised) and stores them in pgvector (HNSW index) alongside a full-text GIN index.

### Question answering

`POST /ask` runs a LangGraph `StateGraph`:

```
decompose → retrieve → grade ─┬─ ≥2 relevant chunks ─────────→ reason → format → END
                ↑             ├─ <2 relevant, no retry yet ──→ retry → augment ─┐
                └─────────────┼─────────────────────────────────────────────────┘
                              ├─ after retry, ≥1 relevant ───→ reason → format → END
                              └─ after retry, 0 relevant ────→ clarify → END
```

| Node | What it does |
|---|---|
| `decompose` | LLM splits the question into 2–3 sub-questions, using conversation history for follow-ups |
| `retrieve` | Hybrid search per sub-question: pgvector cosine + Postgres full-text search (top 10 each), fused with Reciprocal Rank Fusion (k=60), top 6 kept |
| `grade` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each chunk against the question; chunks scoring below 0 are dropped |
| `augment` | On retry, LLM rewrites sub-questions with broader domain vocabulary before retrieving again (max one retry) |
| `reason` | LLM answers from the numbered graded chunks |
| `format` | Attaches citations (URL, source, section) |
| `clarify` | When nothing relevant is found, returns 2–3 "I want to know about…" intent options instead of an answer |

The API is stateless: clients send prior turns in `history` with each request.

Every `/ask` call gets a `run_id` that is bound to all log lines (structlog JSON), passed to LangSmith tracing when enabled, and written to the `query_log` table with latency, route, chunk counts and retry count.

## Stack

Python 3.13 · FastAPI · LangGraph · LangChain · Groq (OpenAI-compatible API) · sentence-transformers · Postgres + pgvector · crawl4ai · arq + Redis · uv

## Getting started

### Prerequisites

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Docker (for local Postgres and Redis)
- A [Groq](https://console.groq.com) API key

### Setup

```bash
uv sync --dev                 # create .venv and install dependencies
uv run crawl4ai-setup         # install Playwright browsers used by crawl4ai
cp .env.example .env          # then fill in GROQ_API_KEY and INGEST_API_KEY
docker compose up -d          # Postgres (pgvector) + Redis
```

On first start, the Postgres container initialises its schema from `db/schema.sql`. For an existing database, apply migrations instead:

```bash
uv run python -m db.migrate        # apply pending migrations
uv run python -m db.migrate down   # roll back the last one
```

### Run

In two terminals:

```bash
uv run uvicorn api.app:app --reload   # API on http://localhost:8000
uv run python -m worker               # ingestion worker
```

### Ingest content

```bash
curl -X POST localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.olevalmis.ee", "mode": "crawl"}'
# → {"job_id": "...", "status": "queued"}

curl localhost:8000/ingest/jobs/<job_id>
```

### Ask a question

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I keep in a 72-hour emergency kit?", "history": []}'
```

Response fields: `answer`, `citations`, `sub_questions`, `clarifying_questions`, `run_id`.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/ask` | Ask a question (rate-limited to 10/min per IP) |
| `POST` | `/ingest/url` | Queue a crawl (`mode`: `crawl` = whole sitemap, `single` = one page) |
| `GET` | `/ingest/jobs/{job_id}` | Ingestion job status and result |
| `GET` | `/documents` | Ingested sources with chunk counts |
| `GET` | `/topics` | Distinct section headings across ingested chunks |
| `GET` | `/health` | Health check |

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq API key |
| `DATABASE_URL` | yes | Postgres connection string (append `?sslmode=require` for Neon/Supabase) |
| `GROQ_MODEL` | yes | Groq model id, e.g. `llama-3.3-70b-versatile` |
| `SYSTEM_PROMPT` | yes | Assistant persona and scope |
| `INGEST_API_KEY` | yes | Key for the ingestion API-key check (`api/deps.py::verify_api_key`) |
| `REDIS_URL` | no | Default `redis://localhost:6379` |
| `ALLOWED_ORIGINS` | no | Comma-separated CORS origins, default `*` |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | no | LangSmith tracing |

## Testing

Integration tests run against a live API with content already ingested:

```bash
uv run pytest tests/ -v
```

LLM-as-judge evaluation (scores relevance, groundedness and completeness 1–5; uses Groq credits):

```bash
uv run python tests/eval/run_eval.py
```

## Deployment

`Dockerfile` builds the API image with Playwright/Chromium; `render.yaml` defines a Render web service. Any Postgres with the `vector` extension works (e.g. Neon).

## Project structure

```
agent/        LangGraph graph, state, embeddings, nodes
api/          FastAPI app, routes, dependencies, schemas, logging
db/           asyncpg pool, repository (all SQL), migration runner, schema snapshot
ingestion/    crawler, chunker, chunk quality filter
services/     ingestion orchestration
worker/       arq background worker
migrations/   numbered SQL migrations
tests/        integration tests and eval harness
```
