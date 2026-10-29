"""Data loading for the dashboard. Streamlit scripts are sync, so each loader runs one short-lived async pool."""
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import asyncpg
from asyncpg import Pool
import numpy as np
import pandas as pd
import streamlit as st

from agent.embed import get_encoder
from db.client import create_pool
from db.eval_repository import EvalRepository
from db.query_log_repository import QueryLogRepository
from db.repository import DocumentRepository
from evaluation.online import score_row
from tests.eval.thresholds import RESULTS_DIR

T = TypeVar("T")


def _run(query: Callable[[Pool], Awaitable[T]]) -> T:
    async def go() -> T:
        pool = await create_pool()
        try:
            return await query(pool)
        finally:
            await pool.close()

    return asyncio.run(go())


def source_label(url: str) -> str:
    """Domain plus last path segment: distinct for both https://a.org/en and https://b.org/en."""
    parsed = urlparse(url)
    last = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return f"{parsed.netloc}/{last}" if last else parsed.netloc


def _frame(rows: list[asyncpg.Record], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows], columns=columns)


@st.cache_data(ttl=60, show_spinner="Loading chunks…")
def load_chunks() -> tuple[pd.DataFrame, np.ndarray]:
    encoder = get_encoder()  # the vectors search uses
    rows = _run(lambda pool: DocumentRepository(pool).chunk_embeddings(encoder.name))
    df = _frame(rows, ["id", "content", "source", "uploaded_at"])
    df["source"] = df["source"].map(source_label)
    df["length"] = df["content"].str.len()
    embeddings = np.array([np.asarray(r["embedding"]) for r in rows]) if rows else np.empty((0, encoder.dims))
    return df, embeddings


@st.cache_data(ttl=60)
def load_query_log(days: int) -> pd.DataFrame:
    columns = ["created_at", "question", "route", "latency_ms", "retry_count",
               "retrieved_chunk_count", "graded_chunk_count"]
    return _frame(_run(lambda pool: QueryLogRepository(pool).query_log_since(days)), columns)


@st.cache_data(ttl=60)
def load_eval_scores(days: int) -> pd.DataFrame:
    columns = ["evaluated_at", "metric", "score", "judge_model", "question", "answer"]
    return _frame(_run(lambda pool: EvalRepository(pool).eval_scores_since(days)), columns)


def load_chats(limit: int, offset: int) -> list[dict[str, Any]]:
    return [dict(r) for r in _run(lambda pool: QueryLogRepository(pool).recent_chats(limit, offset))]


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

    return _run(go)


def score_turn(run_id: str) -> dict[str, float]:
    """Run the judge on one logged answer now; the scores land in query_eval like the nightly ones."""
    async def go(pool: Pool) -> dict[str, float]:
        row = await QueryLogRepository(pool).query_for_scoring(run_id)
        return await score_row(DocumentRepository(pool), EvalRepository(pool), row) if row else {}

    return _run(go)

def result_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*.json"), reverse=True)


@st.cache_data(ttl=60)
def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
