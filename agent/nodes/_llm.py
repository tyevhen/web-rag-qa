import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from settings import get_settings

_HF_ROUTER_HOST = "router.huggingface.co"
# "1. ", "2) ", "- ", "* ", "• " that models add despite "no numbering"
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
# (base_url, model) for the current task only; the dashboard chat sets it to compare models. The API never does.
_override: ContextVar[tuple[str, str] | None] = ContextVar("llm_override", default=None)


@contextmanager
def use_model(base_url: str, model: str) -> Iterator[None]:
    token = _override.set((base_url, model))
    try:
        yield
    finally:
        _override.reset(token)


def get_groq_llm() -> ChatOpenAI:
    s = get_settings()
    return _build_llm(*(_override.get() or (s.llm_base_url, s.groq_model)))


@lru_cache(maxsize=8)
def _build_llm(base_url: str, model: str) -> ChatOpenAI:
    s = get_settings()
    use_hf = _HF_ROUTER_HOST in base_url
    if use_hf and s.hf_token is None:
        raise ValueError("LLM_BASE_URL points at Hugging Face but HF_TOKEN is not set")
    return ChatOpenAI(
        model=model,
        api_key=s.hf_token if use_hf else s.groq_api_key,
        base_url=base_url,
        temperature=0,  # repeatable answers, so eval runs are comparable
        timeout=s.llm_timeout,  # per attempt; the client default (600 s) lets a stalled provider hang /ask
        # Free tiers return 429 on per-minute token limits; the client backs off using retry-after
        max_retries=6,
    )


async def invoke_chain(prompt: ChatPromptTemplate, **kwargs: Any) -> str:
    return await (prompt | get_groq_llm() | StrOutputParser()).ainvoke(kwargs)


def parse_lines(content: str) -> list[str]:
    """One item per non-blank line, list markers removed."""
    # shortcut: a preamble line ("Here are the queries:") is kept as an item; filter it if evals show it happens
    return [item for line in content.splitlines() if (item := _LIST_MARKER.sub("", line).strip())]


async def invoke_lines(prompt: ChatPromptTemplate, **kwargs: Any) -> list[str]:
    return parse_lines(await invoke_chain(prompt, **kwargs))
