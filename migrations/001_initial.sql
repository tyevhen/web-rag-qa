-- migrate:up
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL,
    domain      TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB NOT NULL DEFAULT '{}'
);

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

-- migrate:down
DROP TABLE IF EXISTS query_log;
DROP TABLE IF EXISTS documents;
DROP EXTENSION IF EXISTS vector;
