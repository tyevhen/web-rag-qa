import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data import load_chunks
from dashboard.topics import boilerplate_phrases, cluster, label_clusters, project_2d

st.set_page_config(page_title="Corpus", layout="wide")
st.title("Corpus")
st.caption("What is stored in the vector index. Topics come from clustering the chunk embeddings, with no LLM calls.")

chunks, embeddings = load_chunks()
if chunks.empty:
    st.info("No chunks ingested yet. Use POST /ingest/url.")
    st.stop()

sources = pd.DataFrame(chunks.groupby("source").agg(
    chunks=("id", "count"),
    avg_chars=("length", "mean"),
    max_chars=("length", "max"),
    ingested=("uploaded_at", "min"),
))
sources["avg_chars"] = sources["avg_chars"].round(0)
c1, c2, c3 = st.columns(3)
c1.metric("Chunks", len(chunks))
c2.metric("Sources", len(sources))
lengths = chunks["length"].to_numpy()
c3.metric("Median chunk length", f"{int(np.median(lengths))} chars")
st.dataframe(sources, width="stretch")

st.subheader("Chunk length distribution")
counts, edges = np.histogram(lengths, bins=np.arange(0, int(lengths.max()) + 151, 150))
st.bar_chart(pd.Series(counts, index=[f"{int(a)}–{int(b)}" for a, b in zip(edges[:-1], edges[1:])], name="chunks"),
             x_label="characters", y_label="chunks", sort=False)

st.subheader("Topics")
k = st.slider("Number of topics", min_value=4, max_value=30, value=12)


@st.cache_data(ttl=60, show_spinner="Clustering…")
def topics(k: int, n: int) -> tuple[list[int], dict[int, str], list[list[float]]]:
    labels = cluster(embeddings, k)
    return labels.tolist(), label_clusters(chunks["content"].tolist(), labels), project_2d(embeddings).tolist()


labels, names, coords = topics(k, len(chunks))
chunks["topic"] = [names[label] for label in labels]
summary = (
    chunks.pivot_table(index="topic", columns="source", values="id", aggfunc="count", fill_value=0)
    .assign(total=lambda df: df.sum(axis=1))
    .sort_values("total", ascending=False)
)
st.dataframe(summary, width="stretch")

st.markdown("**Topic map**: each point is a chunk, positioned by a 2D projection of its embedding.")
points = pd.DataFrame(coords, columns=["x", "y"]).assign(topic=chunks["topic"])
st.scatter_chart(points, x="x", y="y", color="topic", height=500)

with st.expander("Sample chunks per topic"):
    topic = st.selectbox("Topic", summary.index.tolist())
    for _, row in chunks[chunks["topic"] == topic].head(3).iterrows():
        st.caption(f"{row['source']} · {row['length']} chars")
        st.text(row["content"][:600])

st.subheader("Boilerplate")
st.caption(
    "Phrases repeated verbatim in at least 30% of one source's chunks: page banners, headers, menus. "
    "They add the same noise to every chunk's embedding and keyword index."
)
found = boilerplate_phrases(chunks["content"].tolist(), chunks["source"].tolist())
rows = [{"source": s, "share of chunks": f"{share:.0%}", "phrase": p} for s, items in found.items() for p, share in items]
if rows:
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
else:
    st.success("No repeated boilerplate found.")
