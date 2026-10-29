"""Chat with the real graph, optionally two models side by side. Every turn is saved to query_log with a chat_id
(kept out of Production stats and nightly judging); the judge runs only when Score is pressed."""
import asyncio
import time
import urllib.request
import uuid
from typing import Any

import streamlit as st

from agent.config import get_agent_config
from agent.graph import build_graph, route_of
from agent.nodes._llm import use_model
from agent.state import AgentState
from dashboard.chat_list import chat_sidebar
from dashboard.data import load_chat, score_turn
from dashboard.views import context_chunks
from db.client import create_pool
from db.query_log_repository import QueryLogEntry, QueryLogRepository
from db.repository import DocumentRepository
from evaluation.judge import judge_model_name

_LOCAL_URL = "http://localhost:11434/v1"
_SCORE_LABELS = {"faithfulness": "sticks to the sources", "answer_relevancy": "answers the question asked"}

st.set_page_config(page_title="Chat", layout="wide")
st.title("Chat")
st.caption("Ask the real pipeline (retrieve → grade → reason) with a model of your choice. Chats are saved. "
           "This page calls the LLM: Groq models count against the free-tier limits.")


@st.cache_data(ttl=30, show_spinner=False)
def ollama_up() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5):
            return True
    except OSError:
        return False


groq_url = "https://api.groq.com/openai/v1"  # not llm_base_url: that may point the app at Ollama or HF
models = {"openai/gpt-oss-120b": ("Groq · gpt-oss-120b", groq_url), "openai/gpt-oss-20b": ("Groq · gpt-oss-20b", groq_url)}
if ollama_up():
    models["web-rag-llm"] = ("Local · web-rag-llm", _LOCAL_URL)
else:
    st.caption("Local model hidden: Ollama is not answering on :11434 (see README, Local LLM).")

current = st.session_state.setdefault("chat_id", str(uuid.uuid4()))

c1, c2 = st.columns(2)
first = c1.selectbox("Model", list(models), format_func=lambda m: models[m][0])
second = c2.selectbox("Compare with", [None, *[m for m in models if m != first]],
                      format_func=lambda m: "—" if m is None else models[m][0])
chosen = [m for m in (first, second) if m]


async def _one(graph: Any, model: str, question: str, history: list[dict]) -> QueryLogEntry:
    initial: AgentState = {"question": question, "history": history, "sub_questions": [], "retrieved_chunks": [],
                           "graded_chunks": [], "answer": "", "citations": [], "clarifying_questions": [],
                           "retry_count": 0}
    entry = QueryLogEntry(run_id=str(uuid.uuid4()), question=question, answer="", domain=None, latency_ms=0,
                          route="error", sub_question_count=0, retrieved_chunk_count=0, graded_chunk_count=0,
                          retry_count=0, citations=[], graded_chunk_ids=[], chat_id=current, model=model)
    t0 = time.monotonic()
    with use_model(models[model][1], model):
        try:
            state = await graph.ainvoke(initial)
        except Exception as exc:  # record the provider error (rate limit, timeout) instead of crashing the page
            entry.answer, entry.latency_ms = f"Error: {exc}", int((time.monotonic() - t0) * 1000)
            return entry
    graded = state["graded_chunks"]
    entry.answer, entry.route, entry.retry_count = state["answer"], route_of(state), state["retry_count"]
    entry.latency_ms = int((time.monotonic() - t0) * 1000)
    entry.domain = graded[0].metadata.get("source") if graded else None
    entry.sub_questions, entry.sub_question_count = state["sub_questions"], len(state["sub_questions"])
    entry.retrieved_chunk_count, entry.graded_chunk_count = len(state["retrieved_chunks"]), len(graded)
    entry.citations, entry.graded_chunk_ids = state["citations"], [c.id for c in graded]
    return entry


async def _ask(question: str, history: dict[str, list[dict[str, Any]]]) -> None:
    pool = await create_pool()
    try:
        graph = build_graph(get_agent_config(), DocumentRepository(pool))
        entries = await asyncio.gather(*(
            _one(graph, m, question, [{"role": r, "content": t[k]} for t in history.get(m, []) if t["route"] != "error"
                                      for r, k in (("user", "question"), ("assistant", "answer"))])
            for m in chosen))
        log = QueryLogRepository(pool)
        for entry in entries:
            await log.log_query(entry)
    finally:
        await pool.close()


def show(turn: dict[str, Any]) -> None:
    st.chat_message("user").write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"] or "_(no answer)_")
        st.caption(f"route {turn['route']} · retries {turn['retry_count']} · {turn['latency_ms'] / 1000:.1f} s · "
                   f"search queries {len(turn['sub_questions'] or [])} · passages found {turn['retrieved_chunk_count']}"
                   f", kept by filter {turn['graded_chunk_count']} · citations {len(turn['citations'] or [])}")
        if turn["sub_questions"]:
            with st.expander("Search queries"):
                st.write(turn["sub_questions"])
        if turn["citations"]:
            with st.expander("Citations"):
                st.write(turn["citations"])
        if turn["context"]:
            with st.expander(f"Passages the model saw ({len(turn['context'])})"):
                context_chunks(turn["context"])
        if turn["scores"]:
            st.caption("Judge: " + " · ".join(f"{_SCORE_LABELS.get(k, k)} {v:.2f}" for k, v in turn["scores"].items()))
        elif turn["route"] == "reason" and turn["graded_chunk_ids"]:
            if judge_model_name().endswith(f":{turn['model']}"):
                st.caption("The judge is the same model that wrote this answer, so it may favour it.")
            if st.button("Score", key=f"score-{turn['run_id']}"):
                with st.spinner("Judging…"):
                    try:
                        scored = score_turn(turn["run_id"])
                    except Exception as exc:  # e.g. Groq 429 once the daily token limit is used up
                        st.error(f"Judge failed: {exc}")
                        return
                if not scored:
                    st.warning("The judge returned no scores (rate limit or unreadable reply); try again later.")
                    return
                st.rerun()


if question := st.chat_input("Ask a question"):
    with st.spinner(f"Asking {', '.join(models[m][0] for m in chosen)}…"):
        asyncio.run(_ask(question, load_chat(current)))

turns = load_chat(current)
shown = [*chosen, *[m for m in turns if m not in chosen]]
for col, m in zip(st.columns(len(shown)), shown, strict=True):
    with col:
        st.markdown(f"**{models[m][0] if m in models else m}**")
        for turn in turns.get(m, []):
            show(turn)

chat_sidebar(current)  # last, so a chat started on this run is already listed
