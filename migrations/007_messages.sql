-- migrate:up
-- Server-side conversation history for /ask: one row per user or assistant turn
CREATE TABLE IF NOT EXISTS messages (
    conversation_id UUID NOT NULL,
    seq             INT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, seq)
);

-- migrate:down
DROP TABLE IF EXISTS messages;
