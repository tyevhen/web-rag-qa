"""Production traffic (query_log) and its nightly judge scores (query_eval)."""
import pandas as pd
import streamlit as st

from dashboard.data._db import frame, run
from db.repositories.evals import EvalRepository
from db.repositories.query_log import QueryLogRepository


@st.cache_data(ttl=60)
def load_query_log(days: int) -> pd.DataFrame:
    columns = ["created_at", "question", "route", "latency_ms", "retry_count",
               "retrieved_chunk_count", "graded_chunk_count"]
    return frame(run(lambda pool: QueryLogRepository(pool).query_log_since(days)), columns)


@st.cache_data(ttl=60)
def load_eval_scores(days: int) -> pd.DataFrame:
    columns = ["evaluated_at", "metric", "score", "judge_model", "question", "answer"]
    return frame(run(lambda pool: EvalRepository(pool).eval_scores_since(days)), columns)
