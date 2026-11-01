"""RAG dashboard.

    PYTHONPATH=. uv run streamlit run dashboard/app.py
"""
import streamlit as st

from dashboard.data import load_chunks, load_query_log, load_result, result_files

st.set_page_config(page_title="RAG dashboard", layout="wide")
st.title("RAG dashboard")

chunks, _ = load_chunks()
files = result_files()
latest = load_result(files[0]) if files else None
traffic = load_query_log(7)

c1, c2, c3 = st.columns(3)
c1.metric("Chunks indexed", len(chunks), help="Across all ingested sources")
c2.metric("Sources", len(set(chunks["source"])))
c3.metric("Requests (7 days)", len(traffic))

if latest:
    st.markdown(
        f"Latest eval run: **{latest['meta']['timestamp']}**, {latest['meta']['rows']} rows, "
        f"**{len(latest.get('violations', []))}** threshold violations."
    )

st.page_link("pages/1_Corpus.py", label="Corpus: what is stored, topics, boilerplate")
st.page_link("pages/2_Offline_eval.py", label="Offline eval: golden-set runs by pipeline stage")
st.page_link("pages/4_Answers.py", label="Answers: read each golden conversation, its citations, and compare runs")
st.page_link("pages/5_Compare_models.py", label="Compare models: two runs side by side, judge scores, interesting cases")
st.page_link("pages/6_Chat.py", label="Chat: ask the pipeline with any model, two side by side")
st.page_link("pages/3_Production.py", label="Production: live traffic and sampled answer quality")
