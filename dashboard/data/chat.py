"""Chat page I/O: models and answers come from the API, saved chats and judge scores from Postgres."""
import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
from asyncpg import Pool
from httpx_sse import aconnect_sse

from dashboard.analysis.citations import chunk_urls, link_markers
from dashboard.data._db import run
from db.repositories.documents import DocumentRepository
from db.repositories.evals import EvalRepository
from db.repositories.query_log import QueryLogRepository
from evaluation.online import score_row
from settings import get_settings

# (model, event, text) for each "token", "status" or "error" event of one model's answer
OnEvent = Callable[[str, str, str], None]
_TEXT_FIELD = {"token": "text", "status": "status", "error": "detail"}


class ApiUnavailable(Exception):
    pass


def available_models() -> tuple[list[str], str]:
    """The models /ask accepts, and the default."""
    try:
        response = httpx.get(f"{get_settings().api_url}/models", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ApiUnavailable(f"{get_settings().api_url}: {exc}") from exc
    body = response.json()
    return body["models"], body["default"]


def load_chats(limit: int, offset: int) -> list[dict[str, Any]]:
    return [dict(r) for r in run(lambda pool: QueryLogRepository(pool).recent_chats(limit, offset))]


def load_chat(chat_id: str) -> dict[str, list[dict[str, Any]]]:
    """A chat's turns per model, oldest first, with the text of the passages each answer saw, in the order it saw them."""
    async def go(pool: Pool) -> dict[str, list[dict[str, Any]]]:
        turns = [dict(r) for r in await QueryLogRepository(pool).chat_turns(chat_id)]
        ids = [i for t in turns for i in t["graded_chunk_ids"] or []]
        found = {c["id"]: c["content"] for c in await DocumentRepository(pool).get_chunks_by_ids(ids)} if ids else {}
        api_url = get_settings().api_url
        for t in turns:
            t["linked_answer"] = link_markers(t["answer"], chunk_urls(t["citations"] or [], api_url))
            t["context"] = [found.get(i, "(re-ingested since: this passage is no longer in the index)")
                            for i in t["graded_chunk_ids"] or []]
        per_model: dict[str, list[dict[str, Any]]] = {}
        for t in turns:
            per_model.setdefault(t["model"], []).append(t)
        return per_model

    return run(go)


def score_turn(run_id: str) -> dict[str, float]:
    """Run the judge on one logged answer now; the scores land in query_eval like the nightly ones."""
    async def go(pool: Pool) -> dict[str, float]:
        row = await QueryLogRepository(pool).query_for_scoring(run_id)
        return await score_row(DocumentRepository(pool), EvalRepository(pool), row) if row else {}

    return run(go)


async def _stream(client: httpx.AsyncClient, chat_id: str, model: str, question: str, on_event: OnEvent) -> None:
    """One model's answer from /ask/stream; the API saves the turn and its query_log row itself."""
    body = {"question": question, "conversation_id": chat_id, "model": model}
    async with aconnect_sse(client, "POST", "/ask/stream", json=body) as source:
        if source.response.is_error:  # rejected before streaming: a 422 or 429 with a JSON body
            await source.response.aread()
            on_event(model, "error", source.response.text)
            return
        async for event in source.aiter_sse():
            if field := _TEXT_FIELD.get(event.event):
                on_event(model, event.event, json.loads(event.data)[field])


async def _ask(chat_id: str, models: list[str], question: str, on_event: OnEvent) -> None:
    s = get_settings()
    timeout = httpx.Timeout(10, read=s.request_timeout + 30)  # the API ends a run at request_timeout
    async with httpx.AsyncClient(base_url=s.api_url, timeout=timeout) as client:
        await asyncio.gather(*(_stream(client, chat_id, m, question, on_event) for m in models))


def ask(chat_id: str, models: list[str], question: str, on_event: OnEvent) -> None:
    """Ask every given model in parallel through the API, in this chat's conversation, streaming each answer."""
    asyncio.run(_ask(chat_id, models, question, on_event))
