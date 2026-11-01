"""Rendering shared by the eval pages."""
from typing import Any

import streamlit as st


def run_label(run: dict[str, Any]) -> str:
    m = run["meta"]
    search = m.get("embedder", "BAAI/bge-small-en-v1.5")  # runs before multi-encoder support used bge
    return (f"{m['timestamp']} · {m.get('generator', 'model not recorded')} · search: {search} · "
            f"judge: {m['judge']} · {m['rows']} rows")


def scores_line(item: dict[str, Any]) -> str:
    scores = item.get("scores") or {}
    return " · ".join(f"{k} {'n/a' if v is None else f'{v:.2f}'}" for k, v in sorted(scores.items()))


def conversation(item: dict[str, Any], title: str) -> None:
    st.markdown(f"**{title}**")
    if line := scores_line(item):
        st.caption(f"Judge: {line}")
    for turn in item.get("history") or []:
        st.chat_message(turn["role"]).write(turn["content"])
    st.chat_message("user").write(item["question"])
    st.chat_message("assistant").write(item["answer"] or "_(no answer)_")


def context_chunks(context: Any) -> None:
    """The graded chunks the reasoner saw; `[n]` in the answer points to chunk n."""
    if not isinstance(context, list):
        st.caption("Not recorded in this run; rerun `run_eval` to include them.")
    elif not context:
        st.caption("None: the grader kept no chunks.")
    for n, text in enumerate(context or [], 1):
        with st.expander(f"[{n}] {text[:100]}"):
            st.text(text)
