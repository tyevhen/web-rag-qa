from functools import lru_cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from settings import get_settings


@lru_cache(maxsize=1)
def get_groq_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.groq_model,
        api_key=s.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )


async def invoke_chain(prompt: ChatPromptTemplate, **kwargs: Any) -> str:
    return await (prompt | get_groq_llm() | StrOutputParser()).ainvoke(kwargs)
