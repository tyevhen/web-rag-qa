-- Reference snapshot of the current schema.
-- Source of truth: migrations/ (run: uv run python db/migrate.py)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL,
    domain      TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    embedding   vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS chunks_content_fts_gin_idx
    ON chunks USING gin (to_tsvector('english', content));

CREATE TABLE IF NOT EXISTS query_log (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question                TEXT NOT NULL,
    answer                  TEXT NOT NULL,
    domain                  TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id                  UUID,
    latency_ms              INT,
    route                   TEXT,
    sub_question_count      INT,
    retrieved_chunk_count   INT,
    graded_chunk_count      INT,
    retry_count             INT,
    citations               JSONB
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
