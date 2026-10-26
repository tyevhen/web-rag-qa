"""Data loading for the dashboard. Streamlit scripts are sync, so each loader runs one short-lived async pool."""
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import asyncpg
import numpy as np
import pandas as pd
import streamlit as st

from db.client import create_pool
from db.repository import DocumentRepository
from tests.eval.thresholds import RESULTS_DIR

T = TypeVar("T")


def _run(query: Callable[[DocumentRepository], Awaitable[T]]) -> T:
    async def go() -> T:
        pool = await create_pool()
        try:
            return await query(DocumentRepository(pool))
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
    rows = _run(lambda repo: repo.chunk_embeddings())
    df = _frame(rows, ["id", "content", "source", "uploaded_at"])
    df["source"] = df["source"].map(source_label)
    df["length"] = df["content"].str.len()
    embeddings = np.array([np.asarray(r["embedding"]) for r in rows]) if rows else np.empty((0, 384))
    return df, embeddings


@st.cache_data(ttl=60)
def load_query_log(days: int) -> pd.DataFrame:
    columns = ["created_at", "question", "route", "latency_ms", "retry_count",
               "retrieved_chunk_count", "graded_chunk_count"]
    return _frame(_run(lambda repo: repo.query_log_since(days)), columns)


@st.cache_data(ttl=60)
def load_eval_scores(days: int) -> pd.DataFrame:
    columns = ["evaluated_at", "metric", "score", "judge_model", "question", "answer"]
    return _frame(_run(lambda repo: repo.eval_scores_since(days)), columns)


def result_files() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*.json"), reverse=True)


@st.cache_data(ttl=60)
def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
