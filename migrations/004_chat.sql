-- migrate:up
-- Dashboard Chat turns are logged like API requests; chat_id IS NULL marks API traffic
ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS chat_id UUID,
    ADD COLUMN IF NOT EXISTS model TEXT,
    ADD COLUMN IF NOT EXISTS sub_questions TEXT[];

CREATE INDEX IF NOT EXISTS query_log_chat_id_idx ON query_log (chat_id);

-- migrate:down
DROP INDEX IF EXISTS query_log_chat_id_idx;
ALTER TABLE query_log
    DROP COLUMN IF EXISTS sub_questions,
    DROP COLUMN IF EXISTS model,
    DROP COLUMN IF EXISTS chat_id;
