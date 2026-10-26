from typing import Any, cast

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.config import TopicConfig
from agent.nodes import _llm, reasoner
from agent.nodes.augmenter import make_augmenter
from agent.state import AgentState, Chunk


def _state(**overrides: Any) -> AgentState:
    state: AgentState = {"question": "q", "history": [], "sub_questions": [], "retrieved_chunks": [],
                         "graded_chunks": [], "answer": "", "citations": [], "clarifying_questions": [], "retry_count": 0}
    return cast(AgentState, {**state, **overrides})


def test_parse_lines_strips_list_markers_and_blanks():
    content = "\n1. What is A?\n2) What is B?\n\n- dash item\n* star item\n• bullet\n  plain  \n"

    assert _llm.parse_lines(content) == ["What is A?", "What is B?", "dash item", "star item", "bullet", "plain"]


def test_parse_lines_keeps_numbers_inside_text():
    assert _llm.parse_lines("2024 budget\n3.5 percent rate") == ["2024 budget", "3.5 percent rate"]


@pytest.mark.parametrize(("response", "expected"), [
    ("1. a plain\n2. b plain", ["a", "b", "a plain", "b plain"]),  # originals first, then rewrites
    ("a\nb plain", ["a", "b", "b plain"]),                         # unchanged rewrite is not duplicated
    ("only one line", ["a", "b"]),                                  # wrong count: originals only
])
async def test_augmenter(monkeypatch, response: str, expected: list[str]):
    monkeypatch.setattr(_llm, "get_groq_llm", lambda: FakeListChatModel(responses=[response]))

    result = await make_augmenter()(_state(sub_questions=["a", "b"]))

    assert result["sub_questions"] == expected


async def test_reasoner_numbers_context_in_graded_order_and_maps_history(monkeypatch):
    sent: dict[str, Any] = {}

    async def record(prompt, **kwargs):
        sent["messages"] = prompt.format_messages(**kwargs)
        return " answer [1] "

    monkeypatch.setattr(reasoner, "invoke_chain", record)
    chunks = [Chunk(content=t, metadata={}, id=t, score=0.0) for t in ("first", "second")]
    history = [{"role": "user", "content": "hi"}, {"role": "tool", "content": "x"}, {"role": "assistant", "content": "hello"}]

    result = await reasoner.make_reasoner(TopicConfig(system_prompt="persona"))(_state(graded_chunks=chunks, history=history))

    system, user_turn, ai_turn, question = sent["messages"]
    assert result == {"answer": "answer [1]"}
    assert isinstance(system, SystemMessage) and system.content == "persona"
    assert isinstance(user_turn, HumanMessage) and user_turn.content == "hi"  # unknown "tool" role dropped
    assert isinstance(ai_turn, AIMessage) and ai_turn.content == "hello"
    assert "[1] first\n\n---\n\n[2] second" in str(question.content)
