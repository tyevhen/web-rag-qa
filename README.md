# web-rag-qa

Agentic RAG question-answering service. Point it at a website — it crawls the pages, embeds the content into Postgres/pgvector, and answers natural-language questions over it with source citations.

The deployment in this repo is configured as an **emergency preparedness assistant** over public sources on disaster preparedness, survival and tactical casualty care (see [Corpus sources](#corpus-sources)). The domain is set entirely by the `SYSTEM_PROMPT` environment variable — the pipeline itself is topic-agnostic.

## How it works

### Ingestion

`POST /ingest/url` enqueues a background job (arq + Redis). The worker:

1. **Crawls** the site's sitemap with crawl4ai (Playwright-rendered, JS-capable), or a single page in `single` mode.
2. **Chunks** each page on Markdown headings, keeping `url`, `source` and `section` metadata.
3. **Filters** chunks: under 20 words, under 50% alphabetic characters, exact duplicates (MD5) and non-English text (langdetect) are dropped.
4. **Embeds** chunks with `BAAI/bge-small-en-v1.5` (384-dim, L2-normalised; chunks are capped at its 512-token window so none are cut off) and stores them in pgvector (one HNSW index per encoder) alongside a full-text GIN index. See [Switching encoders](#switching-encoders).

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
| `grade` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each chunk against the question; chunks scoring below −1.0 are dropped |
| `augment` | On retry, LLM rewrites sub-questions with broader domain vocabulary before retrieving again (max one retry) |
| `reason` | LLM answers from the numbered graded chunks |
| `format` | Lists the passages behind each `[n]` (section, source page, link to the passage) |
| `clarify` | When nothing relevant is found, returns 2–3 "I want to know about…" intent options instead of an answer |

The server keeps conversation history in the `messages` table: send the `conversation_id` from a response with the next question to continue that conversation (the last 10 turns are used). Answered and clarify turns are stored; blocked, timed-out and failed ones are not.

Every `/ask` call gets a `run_id` that is bound to all log lines (structlog JSON), passed to LangSmith tracing when enabled, and written to the `query_log` table with latency, route, chunk counts and retry count.

## Stack

Python 3.13 · FastAPI · LangGraph · LangChain · Ollama (local, OpenAI-compatible API) · sentence-transformers · Postgres + pgvector · crawl4ai · arq + Redis · uv

## Getting started

### Prerequisites

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Docker (for local Postgres and Redis)
- [Ollama](https://ollama.com) with the app model (see Local LLM)
- For judged evals only: a key for an OpenAI-compatible endpoint with a free tier

### Setup

```bash
uv sync --dev                 # create .venv and install dependencies
uv run crawl4ai-setup         # install Playwright browsers used by crawl4ai
cp .env.example .env          # then fill in INGEST_API_KEY (and the EVAL_JUDGE_* lines for judged evals)
docker compose up -d          # Postgres (pgvector) + Redis
```

The schema comes only from `migrations/`, on a fresh database too:

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

#### Corpus sources

The deployed corpus is exactly these four ingests. Re-run them against an empty database to rebuild it:

| URL | `mode` | Content |
|---|---|---|
| `https://www.olevalmis.ee/en` | `crawl` | Estonian disaster preparedness guidance |
| `https://tccc.org.ua/en` | `crawl` | Tactical Combat Casualty Care (Ukrainian project, English section) |
| `https://archive.org/download/Fm21-76SurvivalManual/FM21-76_SurvivalManual_djvu.txt` | `single` | US Army FM 21-76 Survival Manual (OCR text) |
| `https://archive.org/download/FM4-25.11/FM4-25.11_djvu.txt` | `single` | US Army FM 4-25.11 First Aid (OCR text) |

Use the archive.org `/download/` plain-text URLs, not `/stream/` pages: the latter wrap the text in archive.org's site menu.

### Ask a question

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I keep in a 72-hour emergency kit?"}'
```

Response fields: `answer`, `citations`, `sub_questions`, `clarifying_questions`, `run_id`, `conversation_id`, `model`.

The answer keeps plain `[n]` markers. Each citation is an object: `n`, `chunk_id`, `section`, `source_url` (the page, with a text fragment that scrolls to the passage in Chrome, Edge and Safari) and `url`, which opens the stored passage (`GET /chunks/{id}`: JSON, or a readable page in a browser).

Pick a model with `"model"`: `GET /models` lists the app's model (the default) and the judge endpoint's. Each model keeps its own history in a conversation.

Ask a follow-up in the same conversation by sending the returned `conversation_id`:

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long should each of those supplies last?", "conversation_id": "<conversation_id from the first response>"}'
```

Unknown fields such as the old `history` are rejected with 422.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/ask` | Ask a question (rate-limited to 10/min per IP) |
| `POST` | `/ask/stream` | Same, as server-sent events: `meta`, the answer as `token`s (or `status: held` for sensitive questions), then `done` or `error` |
| `POST` | `/ingest/url` | Queue a crawl (`mode`: `crawl` = whole sitemap, `single` = one page) |
| `GET` | `/ingest/jobs/{job_id}` | Ingestion job status and result |
| `GET` | `/documents` | Ingested sources with chunk counts |
| `GET` | `/health` | Health check |

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | yes | Postgres connection string (append `?sslmode=require` for Neon/Supabase) |
| `LLM_MODEL` | yes | Model id at `LLM_BASE_URL`, e.g. `web-rag-llm` on Ollama |
| `SYSTEM_PROMPT` | yes | Assistant persona and scope |
| `INGEST_API_KEY` | yes | Key for the ingestion API-key check (`api/deps.py::verify_api_key`) |
| `REDIS_URL` | no | Default `redis://localhost:6379` |
| `ALLOWED_ORIGINS` | no | Comma-separated CORS origins, default `*` |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | no | LangSmith tracing |
| `LLM_BASE_URL`, `LLM_API_KEY` | no | OpenAI-compatible endpoint, default local Ollama (no key) |
| `LLM_TIMEOUT`, `REQUEST_TIMEOUT` | no | Seconds per LLM attempt (default 180) and per /ask run (default 600) |
| `EVAL_JUDGE_BASE_URL`, `EVAL_JUDGE_API_KEY`, `EVAL_JUDGE_MODEL` | no | The judge for evals and nightly scoring: any OpenAI-compatible endpoint, a different model from `LLM_MODEL` |
| `EMBEDDER` | no | Encoder search uses (`agent/embed.py` `ENCODERS`), default `BAAI/bge-small-en-v1.5` |
| `SHADOW_EMBEDDERS` | no | JSON list of extra encoders ingestion also writes, e.g. `'["intfloat/multilingual-e5-small"]'` |

### Switching encoders

Vectors from different encoders are not comparable, so a query must be encoded by the same model as the passages it searches. Each chunk can hold one vector per encoder (`chunk_embeddings` table), which allows a switch without downtime or a re-crawl:

1. Add the encoder to `ENCODERS` in `agent/embed.py` and to `SHADOW_EMBEDDERS`; deploy. New ingestions now write both.
2. `uv run python -m embeddings backfill <name>` — vectors for the older chunks, then its search index.
3. Set `EMBEDDER=<name>` and restart. The API refuses to start if any chunk lacks a vector for it.
4. Remove the old encoder from `SHADOW_EMBEDDERS`, then `uv run python -m embeddings drop <old name>`.

`uv run python -m embeddings status` shows how many chunks each encoder covers. To compare two encoders on the same passages, backfill the second one and run the eval with `EMBEDDER=<name>` set for that run; the run records which encoder searched. Chunks are sized with a fixed tokenizer (bge-small, 512 tokens), whatever the search encoder, so an encoder with a smaller window sees long chunks cut off.

### Local LLM

The app generates answers with a local open-weights model through [Ollama](https://ollama.com), which serves the same OpenAI-compatible API. On macOS run it natively: Docker on a Mac cannot use the GPU.

```bash
brew install ollama && brew services start ollama   # background service on :11434, starts at login
ollama create web-rag-llm -f llm/Modelfile           # pulls the base model (~4.7 GB) and pins the context size
```

`LLM_MODEL=web-rag-llm` in `.env` is all it needs. To change the model, edit `FROM` in `llm/Modelfile` and run `ollama create` again. Any other OpenAI-compatible endpoint works through `LLM_BASE_URL` and `LLM_API_KEY`.

Expect answers in tens of seconds on an M1 laptop. The service log is `$(brew --prefix)/var/log/ollama.log`; a "truncating input prompt" line there means `num_ctx` is too small.

## Testing

Integration tests run against a live API with content already ingested:

```bash
uv run pytest tests/ -v
```

LLM-as-judge evaluation (the judge at `EVAL_JUDGE_BASE_URL`; a free tier's token limits apply):

```bash
uv run python tests/eval/run_eval.py
```

## Deployment

`Dockerfile` builds the API image with Playwright/Chromium; `render.yaml` defines a Render web service. Any Postgres with the `vector` extension works (e.g. Neon).

## Project structure

```
agent/        LangGraph graph, state, embeddings, nodes
api/          FastAPI app, routes, dependencies, schemas, logging
db/           asyncpg pool, repositories (all SQL, one per table area), migration runner, schema snapshot
embeddings/   per-encoder vector upkeep and its CLI (status, backfill, drop)
ingestion/    crawler, chunker, chunk quality filter, ingestion pipeline
worker/       arq background worker
migrations/   numbered SQL migrations
tests/        integration tests and eval harness
```
