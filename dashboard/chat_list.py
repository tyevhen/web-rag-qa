"""Saved chats in the sidebar, most recently active first, one page at a time; clicking a chat opens it."""
import uuid

import streamlit as st

from dashboard.data import load_chats

_PAGE_SIZE = 10


def _open(chat_id: str) -> None:
    st.session_state["chat_id"] = chat_id


def _turn_page(step: int) -> None:
    st.session_state["chat_page"] += step


def chat_sidebar(current: str) -> None:
    """Clicks only set session state; Streamlit then reruns the page with the chosen chat."""
    page = st.session_state.setdefault("chat_page", 0)
    chats = load_chats(_PAGE_SIZE + 1, page * _PAGE_SIZE)  # the extra row only tells whether an older page exists
    with st.sidebar:
        st.button("New chat", icon=":material/add:", width="stretch", on_click=_open, args=(str(uuid.uuid4()),))
        st.subheader("Saved chats")
        if not chats:
            st.caption("No chats on this page yet. Ask a question to start one." if page == 0 else "No older chats.")
        for c in chats[:_PAGE_SIZE]:
            st.button(c["title"], key=f"chat-{c['chat_id']}", width="stretch", on_click=_open, args=(c["chat_id"],),
                      type="primary" if c["chat_id"] == current else "secondary")
            models = ", ".join(m.split("/")[-1] for m in c["models"] if m)
            st.caption(f"{c['started'].astimezone():%b %d %H:%M} · {c['turns']} answers · {models}")
        # Icon-only: worded buttons wrap in the narrow sidebar; the tooltip carries the words
        newer, where, older = st.columns([1, 2, 1], vertical_alignment="center")
        newer.button(":material/chevron_left:", help="Newer chats", disabled=page == 0, on_click=_turn_page, args=(-1,))
        where.caption(f"Page {page + 1}", text_alignment="center")
        older.button(":material/chevron_right:", help="Older chats", disabled=len(chats) <= _PAGE_SIZE,
                     on_click=_turn_page, args=(1,))
