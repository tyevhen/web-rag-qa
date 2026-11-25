import uuid

from asyncpg import Pool


class MessageRepository:
    """The messages table: server-side /ask conversation history, one row per user or assistant turn and model."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def last_turns(self, conversation_id: str, model: str, n: int = 10) -> list[dict]:
        """One model's n most recent turns, oldest first, as {"role", "content"}; [] for an unknown id."""
        rows = await self._pool.fetch(
            "SELECT role, content FROM messages WHERE conversation_id = $1 AND model = $2 ORDER BY seq DESC LIMIT $3",
            uuid.UUID(conversation_id),
            model,
            n,
        )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    async def append(self, conversation_id: str, model: str, question: str, answer: str) -> None:
        """Store a user turn and its answer as the model's next two seq values.

        One statement, so both rows land or neither does. Two concurrent appends for one conversation and model
        read the same max(seq); the second then fails on the primary key instead of interleaving.
        """
        await self._pool.execute(
            """
            INSERT INTO messages (conversation_id, model, seq, role, content)
            SELECT $1, $2, m.next + t.offs, t.role, t.content
            FROM (SELECT coalesce(max(seq), 0) + 1 AS next FROM messages WHERE conversation_id = $1 AND model = $2) m,
                 (VALUES (0, 'user', $3::text), (1, 'assistant', $4::text)) AS t(offs, role, content)
            """,
            uuid.UUID(conversation_id),
            model,
            question,
            answer,
        )
