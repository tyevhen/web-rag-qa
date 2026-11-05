"""Stored chunks and the topics found in them."""
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st

from agent.embed import get_encoder
from dashboard.analysis.topics import cluster, label_clusters, project_2d
from dashboard.data._db import frame, run
from db.repositories.documents import DocumentRepository


def source_label(url: str) -> str:
    """Domain plus last path segment: distinct for both https://a.org/en and https://b.org/en."""
    parsed = urlparse(url)
    last = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return f"{parsed.netloc}/{last}" if last else parsed.netloc


@st.cache_data(ttl=60, show_spinner="Loading chunks…")
def load_chunks() -> tuple[pd.DataFrame, np.ndarray]:
    encoder = get_encoder()  # the vectors search uses
    rows = run(lambda pool: DocumentRepository(pool).chunk_embeddings(encoder.name))
    df = frame(rows, ["id", "content", "source", "uploaded_at"])
    df["source"] = df["source"].map(source_label)
    df["length"] = df["content"].str.len()
    embeddings = np.array([np.asarray(r["embedding"]) for r in rows]) if rows else np.empty((0, encoder.dims))
    return df, embeddings


@st.cache_data(ttl=60, show_spinner="Clustering…")
def load_topics(k: int) -> tuple[list[int], dict[int, str], list[list[float]]]:
    """Topic id per chunk, topic names, and 2D map coordinates, in load_chunks() row order."""
    chunks, embeddings = load_chunks()
    labels = cluster(embeddings, k)
    return labels.tolist(), label_clusters(chunks["content"].tolist(), labels), project_2d(embeddings).tolist()
