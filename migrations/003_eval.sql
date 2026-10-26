-- migrate:up
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS graded_chunk_ids UUID[];

CREATE TABLE IF NOT EXISTS query_eval (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id       UUID NOT NULL,
    metric       TEXT NOT NULL,
    score        REAL,
    judge_model  TEXT NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS query_eval_run_id_idx ON query_eval (run_id);
CREATE INDEX IF NOT EXISTS query_log_created_at_idx ON query_log (created_at);

-- migrate:down
DROP INDEX IF EXISTS query_log_created_at_idx;
DROP INDEX IF EXISTS query_eval_run_id_idx;
DROP TABLE IF EXISTS query_eval;
ALTER TABLE query_log DROP COLUMN IF EXISTS graded_chunk_ids;
