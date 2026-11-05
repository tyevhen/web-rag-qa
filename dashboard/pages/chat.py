"""Chat with the real graph, optionally two models side by side. Every turn is saved to query_log with a chat_id
(kept out of Production stats and nightly judging); the judge runs only when Score is pressed."""
import uuid
from typing import Any

import streamlit as st

from dashboard.components.chat_list import PAGE_SIZE, chat_sidebar
from dashboard.components.conversation import context_chunks
from dashboard.data.chat import ask, available_models, load_chat, load_chats, score_turn
from evaluation.judge import judge_model_name

_SCORE_LABELS = {"faithfulness": "sticks to the sources", "answer_relevancy": "answers the question asked"}

st.title("Chat")
st.caption("Ask the real pipeline (retrieve → grade → reason) with a model of your choice. Chats are saved. "
           "This page calls the LLM: Groq models count against the free-tier limits.")

models, local = available_models()
if not local:
    st.caption("Local model hidden: Ollama is not answering on :11434 (see README, Local LLM).")

current = st.session_state.setdefault("chat_id", str(uuid.uuid4()))

c1, c2 = st.columns(2)
first = c1.selectbox("Model", list(models), format_func=lambda m: models[m][0])
second = c2.selectbox("Compare with", [None, *[m for m in models if m != first]],
                      format_func=lambda m: "—" if m is None else models[m][0])
chosen = [m for m in (first, second) if m]


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
        ask(current, {m: models[m] for m in chosen}, question)

turns = load_chat(current)
shown = [*chosen, *[m for m in turns if m not in chosen]]
for col, m in zip(st.columns(len(shown)), shown, strict=True):
    with col:
        st.markdown(f"**{models[m][0] if m in models else m}**")
        for turn in turns.get(m, []):
            show(turn)

# last, so a chat started on this run is already listed
page = st.session_state.setdefault("chat_page", 0)
chat_sidebar(current, load_chats(PAGE_SIZE + 1, page * PAGE_SIZE), page)
