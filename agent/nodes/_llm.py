import re
from functools import lru_cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ensure_config
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from settings import get_settings

_MAX_QUERIES = 3
_MAX_QUERY_CHARS = 300
# "1. ", "2) ", "- ", "* ", "• " that models add despite "no numbering"
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


def models() -> dict[str, str]:
    """The models a request may pick, as model -> base URL: the app's own (first, the default), and the judge's."""
    s = get_settings()
    picks = {s.llm_model: s.llm_base_url}
    if s.eval_judge_base_url:
        picks.setdefault(s.eval_judge_model, s.eval_judge_base_url)
    return picks


def get_llm() -> ChatOpenAI:
    """The configured model, or the one this run picked: run_graph passes it as configurable["model"],
    and LangGraph hands each node the run's config."""
    s = get_settings()
    model = ensure_config().get("configurable", {}).get("model") or s.llm_model
    return _build_llm(models().get(model, s.llm_base_url), model)


@lru_cache(maxsize=8)
def _build_llm(base_url: str, model: str) -> ChatOpenAI:
    s = get_settings()
    # A request may pick the judge's model, whose endpoint has its own key; Ollama ignores the key
    key = s.eval_judge_api_key if base_url == s.eval_judge_base_url else s.llm_api_key
    return ChatOpenAI(
        model=model,
        api_key=key or SecretStr("unused"),
        base_url=base_url,
        temperature=0,
        timeout=s.llm_timeout,
        max_retries=2,
        # Streamed calls let ServingStats time the first token and let /ask/stream forward the answer as it comes
        streaming=True,
        stream_usage=True,
    )


async def invoke_chain(prompt: ChatPromptTemplate, max_tokens: int | None = None, **kwargs: Any) -> str:
    llm = get_llm()
    # Sent as raw max_tokens: langchain renames the field to max_completion_tokens, which Ollama ignores
    model = llm if max_tokens is None else llm.bind(extra_body={"max_tokens": max_tokens})
    return await (prompt | model | StrOutputParser()).ainvoke(kwargs)


def parse_lines(content: str) -> list[str]:
    """One item per non-blank line, list markers removed."""
    return [item for line in content.splitlines() if (item := _LIST_MARKER.sub("", line).strip())]


async def invoke_lines(prompt: ChatPromptTemplate, max_tokens: int | None = None, **kwargs: Any) -> list[str]:
    return parse_lines(await invoke_chain(prompt, max_tokens, **kwargs))


def cap_queries(queries: list[str]) -> list[str]:
    """At most 3 search queries of 300 chars each."""
    return [q[:_MAX_QUERY_CHARS] for q in queries[:_MAX_QUERIES]]

