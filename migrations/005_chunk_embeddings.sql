-- migrate:up
-- One vector per (chunk, encoder), so a new encoder can be backfilled and compared before search switches to it.
-- The column has no fixed size; each encoder gets a partial index over its own size.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id  UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    encoder   TEXT NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (chunk_id, encoder)
);

-- Assumes the existing vectors are bge-small's (the encoder since the 512-token chunker)
INSERT INTO chunk_embeddings (chunk_id, encoder, embedding)
SELECT id, 'BAAI/bge-small-en-v1.5', embedding FROM chunks
ON CONFLICT DO NOTHING;

CREATE INDEX IF NOT EXISTS emb_baai_bge_small_en_v1_5_hnsw ON chunk_embeddings
    USING hnsw ((embedding::vector(384)) vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE encoder = 'BAAI/bge-small-en-v1.5';

DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;

-- migrate:down
-- Only the bge-small vectors return; chunks without one keep a NULL embedding
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(384);
UPDATE chunks c SET embedding = e.embedding::vector(384)
FROM chunk_embeddings e
WHERE e.chunk_id = c.id AND e.encoder = 'BAAI/bge-small-en-v1.5';
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
DROP TABLE IF EXISTS chunk_embeddings;
