-- migrate:up
-- One conversation can hold turns from several models (the dashboard Chat compares two side by side);
-- each model sees only its own turns. Rows from before this have model '' and drop out of every history.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS model TEXT NOT NULL DEFAULT '';
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_pkey;
ALTER TABLE messages ADD PRIMARY KEY (conversation_id, model, seq);

-- Chat now asks through the API, so every row has a conversation; old Chat rows keep their chat id as it
ALTER TABLE query_log RENAME COLUMN chat_id TO conversation_id;
ALTER INDEX IF EXISTS query_log_chat_id_idx RENAME TO query_log_conversation_id_idx;

-- migrate:down
ALTER INDEX IF EXISTS query_log_conversation_id_idx RENAME TO query_log_chat_id_idx;
ALTER TABLE query_log RENAME COLUMN conversation_id TO chat_id;
DELETE FROM messages WHERE model <> '';
ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_pkey;
ALTER TABLE messages ADD PRIMARY KEY (conversation_id, seq);
ALTER TABLE messages DROP COLUMN IF EXISTS model;
