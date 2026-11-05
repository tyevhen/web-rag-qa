"""Chat page I/O: the models on offer, running the graph per model, and saved chats in query_log."""
import asyncio
import time
import urllib.request
import uuid
from typing import Any

from asyncpg import Pool
import streamlit as st

from agent.config import get_agent_config
from agent.graph import build_graph, log_entry
from agent.nodes._llm import use_model
from agent.state import initial_state
from dashboard.data._db import run
from db.client import create_pool
from db.repositories.documents import DocumentRepository
from db.repositories.evals import EvalRepository
from db.repositories.query_log import QueryLogEntry, QueryLogRepository
from evaluation.online import score_row
from settings import GROQ_BASE_URL

_LOCAL_URL = "http://localhost:11434/v1"

Models = dict[str, tuple[str, str]]  # model id -> (label, base URL)


@st.cache_data(ttl=30, show_spinner=False)
def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5):
            return True
    except OSError:
        return False


def available_models() -> tuple[Models, bool]:
    """The models to offer, and whether the local Ollama one is among them."""
    # Groq, not llm_base_url: that may point the app at Ollama or HF
    models = {"openai/gpt-oss-120b": ("Groq · gpt-oss-120b", GROQ_BASE_URL),
              "openai/gpt-oss-20b": ("Groq · gpt-oss-20b", GROQ_BASE_URL)}
    local = _ollama_up()
    if local:
        models["web-rag-llm"] = ("Local · web-rag-llm", _LOCAL_URL)
    return models, local


def load_chats(limit: int, offset: int) -> list[dict[str, Any]]:
    return [dict(r) for r in run(lambda pool: QueryLogRepository(pool).recent_chats(limit, offset))]


def load_chat(chat_id: str) -> dict[str, list[dict[str, Any]]]:
    """A chat's turns per model, oldest first, with the text of the passages each answer saw, in the order it saw them."""
    async def go(pool: Pool) -> dict[str, list[dict[str, Any]]]:
        turns = [dict(r) for r in await QueryLogRepository(pool).chat_turns(chat_id)]
        ids = [i for t in turns for i in t["graded_chunk_ids"] or []]
        found = {c["id"]: c["content"] for c in await DocumentRepository(pool).get_chunks_by_ids(ids)} if ids else {}
        for t in turns:
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


async def _one(graph: Any, chat_id: str, model: str, base_url: str, question: str,
               history: list[dict]) -> QueryLogEntry:
    run_id, t0 = str(uuid.uuid4()), time.monotonic()
    with use_model(base_url, model):
        try:
            state = await graph.ainvoke(initial_state(question, history))
        except Exception as exc:  # record the provider error (rate limit, timeout) instead of crashing the page
            entry = log_entry(run_id, question, initial_state(question, history), int((time.monotonic() - t0) * 1000),
                              chat_id, model)
            entry.answer, entry.route = f"Error: {exc}", "error"
            return entry
    return log_entry(run_id, question, state, int((time.monotonic() - t0) * 1000), chat_id, model)


def _history(turns: list[dict[str, Any]]) -> list[dict]:
    """One model's earlier turns as graph history; errored turns are left out."""
    return [{"role": r, "content": t[k]} for t in turns if t["route"] != "error"
            for r, k in (("user", "question"), ("assistant", "answer"))]


async def _ask(chat_id: str, models: Models, question: str, history: dict[str, list[dict[str, Any]]]) -> None:
    pool = await create_pool()
    try:
        graph = build_graph(get_agent_config(), DocumentRepository(pool))
        entries = await asyncio.gather(*(
            _one(graph, chat_id, m, url, question, _history(history.get(m, []))) for m, (_, url) in models.items()))
        log = QueryLogRepository(pool)
        for entry in entries:
            await log.log_query(entry)
    finally:
        await pool.close()


def ask(chat_id: str, models: Models, question: str) -> None:
    """Ask every given model in parallel, each with its own history in this chat, and save each answer."""
    asyncio.run(_ask(chat_id, models, question, load_chat(chat_id)))
