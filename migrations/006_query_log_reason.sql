-- migrate:up
-- Why a request did not end in an answer (blocked, timeout, error, rejected); NULL for answered and clarify rows
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS reason TEXT;

-- migrate:down
ALTER TABLE query_log DROP COLUMN IF EXISTS reason;
