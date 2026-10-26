from functools import lru_cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from settings import get_settings

_HF_ROUTER_HOST = "router.huggingface.co"


@lru_cache(maxsize=1)
def get_groq_llm() -> ChatOpenAI:
    s = get_settings()
    use_hf = _HF_ROUTER_HOST in s.llm_base_url
    if use_hf and s.hf_token is None:
        raise ValueError("LLM_BASE_URL points at Hugging Face but HF_TOKEN is not set")
    return ChatOpenAI(
        model=s.groq_model,
        api_key=s.hf_token if use_hf else s.groq_api_key,
        base_url=s.llm_base_url,
        temperature=0,  # repeatable answers, so eval runs are comparable
        timeout=30,  # per attempt; the client default (600 s) lets a stalled provider hang /ask
        # Free tiers return 429 on per-minute token limits; the client backs off using retry-after
        max_retries=6,
    )


async def invoke_chain(prompt: ChatPromptTemplate, **kwargs: Any) -> str:
    return await (prompt | get_groq_llm() | StrOutputParser()).ainvoke(kwargs)
