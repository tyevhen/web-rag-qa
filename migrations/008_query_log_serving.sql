-- migrate:up
-- Serving numbers per LLM call (agent/serving.py: node, time to first token, total time, tokens, time per token)
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS llm_calls JSONB NOT NULL DEFAULT '[]';
-- The answer went to the client as it was written (/ask/stream); a streamed blocked_output row is a leak
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS streamed BOOLEAN NOT NULL DEFAULT false;

-- migrate:down
ALTER TABLE query_log DROP COLUMN IF EXISTS streamed;
ALTER TABLE query_log DROP COLUMN IF EXISTS llm_calls;
