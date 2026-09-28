-- migrate:up
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

-- migrate:down
DROP INDEX IF EXISTS chunks_content_fts_gin_idx;
DROP INDEX IF EXISTS chunks_document_id_idx;
DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;
DROP TABLE IF EXISTS chunks;
