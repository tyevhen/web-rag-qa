"""Chat through the API's /ask/stream, optionally two models side by side in one conversation. The API saves every
turn like any other request; the judge runs only when Score is pressed."""
import uuid
from typing import Any

import streamlit as st

from dashboard.analysis.runs import serving_line
from dashboard.components.chat_list import PAGE_SIZE, chat_sidebar
from dashboard.components.conversation import context_chunks
from dashboard.data.chat import ApiUnavailable, ask, available_models, load_chat, load_chats, score_turn
from evaluation.judge import judge_model_name

_SCORE_LABELS = {"faithfulness": "sticks to the sources", "answer_relevancy": "answers the question asked"}

st.title("Chat")
st.caption("Ask the API (retrieve → grade → reason) with a model of your choice. Chats are saved. "
           "A hosted model counts against its free-tier limits.")

try:
    models, default = available_models()
except ApiUnavailable as exc:
    st.error(f"The API is not answering ({exc}). Start it with `uvicorn api.app:app --reload`.")
    st.stop()


def label(model: str | None) -> str:
    if model is None:
        return "—"
    if model == default:
        return f"App · {model}"
    return f"Judge endpoint · {model}"


current = st.session_state.setdefault("chat_id", str(uuid.uuid4()))

c1, c2 = st.columns(2)
first = c1.selectbox("Model", models, format_func=label)
second = c2.selectbox("Compare with", [None, *[m for m in models if m != first]], format_func=label)
chosen = [m for m in (first, second) if m]


def show(turn: dict[str, Any]) -> None:
    st.chat_message("user").write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["linked_answer"] or "_(no answer)_")
        if turn["reason"]:  # set only for answers that were blocked, timed out or failed
            st.caption(f"Why: {turn['reason']}")
        st.caption(f"route {turn['route']} · retries {turn['retry_count']} · {turn['latency_ms'] / 1000:.1f} s · "
                   f"search queries {len(turn['sub_questions'] or [])} · passages found {turn['retrieved_chunk_count']}"
                   f", kept by filter {turn['graded_chunk_count']} · citations {len(turn['citations'] or [])}")
        if serving := serving_line(turn["llm_calls"]):
            st.caption(serving)
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
                    except Exception as exc:  # e.g. a 429 once the judge's daily token limit is used up
                        st.error(f"Judge failed: {exc}")
                        return
                if not scored:
                    st.warning("The judge returned no scores (rate limit or unreadable reply); try again later.")
                    return
                st.rerun()


question = st.chat_input("Ask a question")
turns = load_chat(current)
shown = [*chosen, *[m for m in turns if m not in chosen]]
replies: dict[str, Any] = {}  # model -> the placeholder its answer streams into
for col, m in zip(st.columns(len(shown)), shown, strict=True):
    with col:
        st.markdown(f"**{label(m)}**")
        for turn in turns.get(m, []):
            show(turn)
        if question and m in chosen:
            st.chat_message("user").write(question)
            replies[m] = st.chat_message("assistant").empty()

if question:
    texts = dict.fromkeys(chosen, "")

    def on_event(model: str, event: str, text: str) -> None:
        if event == "token":
            texts[model] += text
            replies[model].markdown(texts[model])
        elif event == "status":
            replies[model].caption("Held for the safety check; the answer comes whole.")
        else:
            replies[model].error(text)

    ask(current, chosen, question, on_event)
    st.rerun()  # show the saved turns, with their metrics, instead of the streamed text

# last, so a chat started on this run is already listed
page = st.session_state.setdefault("chat_page", 0)
chat_sidebar(current, load_chats(PAGE_SIZE + 1, page * PAGE_SIZE), page)
