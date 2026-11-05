"""RAG dashboard entry: page order, titles and URLs. Pages render only; data/ does the I/O, analysis/ the logic.

    PYTHONPATH=. uv run streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(page_title="RAG dashboard", layout="wide")
st.navigation([
    st.Page("pages/home.py", title="Home", default=True),
    st.Page("pages/corpus.py", title="Corpus"),
    st.Page("pages/offline_eval.py", title="Offline eval"),
    st.Page("pages/production.py", title="Production"),
    st.Page("pages/answers.py", title="Answers"),
    st.Page("pages/compare_models.py", title="Compare models"),
    st.Page("pages/chat.py", title="Chat"),
]).run()
