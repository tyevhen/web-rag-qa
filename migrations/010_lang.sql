-- migrate:up
-- Every chunk before this was ingested English-only
UPDATE chunks SET metadata = metadata || '{"lang": "en"}' WHERE NOT metadata ? 'lang';
-- No estonian text-search config ships with Postgres 16, so Estonian is matched word for word ('simple')
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS fts tsvector GENERATED ALWAYS AS (
    to_tsvector(CASE metadata->>'lang' WHEN 'et' THEN 'simple'::regconfig ELSE 'english'::regconfig END, content)
) STORED;
CREATE INDEX IF NOT EXISTS chunks_fts_gin_idx ON chunks USING gin (fts);
DROP INDEX IF EXISTS chunks_content_fts_gin_idx;
-- The question's language: retrieval searches only chunks in it
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS lang TEXT;

-- migrate:down
ALTER TABLE query_log DROP COLUMN IF EXISTS lang;
CREATE INDEX IF NOT EXISTS chunks_content_fts_gin_idx ON chunks USING gin (to_tsvector('english', content));
DROP INDEX IF EXISTS chunks_fts_gin_idx;
ALTER TABLE chunks DROP COLUMN IF EXISTS fts;
