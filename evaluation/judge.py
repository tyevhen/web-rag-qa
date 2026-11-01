from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import RunConfig
from ragas.embeddings import BaseRagasEmbeddings, LangchainEmbeddingsWrapper
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper

from agent.embed import ENCODERS, embed_documents
from settings import get_settings

_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "hf": "https://router.huggingface.co/v1",
}

# Sequential + patient retries: Groq free tier allows ~8k tokens/min, and 429s surface as NaN scores
RUN_CONFIG = RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=180)


_JUDGE_ENCODER = ENCODERS["BAAI/bge-small-en-v1.5"]


class LocalEmbeddings(Embeddings):
    """LangChain adapter over the app's own bge-small embedder — local, no API cost."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Pinned, not Settings.embedder: comparing search encoders must not change how the judge scores
        return embed_documents(texts, _JUDGE_ENCODER)

    def embed_query(self, text: str) -> list[float]:
        # No query prefix: answer_relevancy compares question to question, so both sides must embed alike
        return embed_documents([text], _JUDGE_ENCODER)[0]


def judge_model_name() -> str:
    s = get_settings()
    return f"{s.eval_judge_provider}:{s.eval_judge_model}"


@lru_cache(maxsize=1)
def get_judge_chat() -> ChatOpenAI:
    s = get_settings()
    if s.eval_judge_provider == "hf" and s.hf_token is None:
        raise ValueError("EVAL_JUDGE_PROVIDER=hf requires HF_TOKEN")
    return ChatOpenAI(
        model=s.eval_judge_model,
        api_key=s.hf_token if s.eval_judge_provider == "hf" else s.groq_api_key,
        base_url=_BASE_URLS[s.eval_judge_provider],
        temperature=0,
        # gpt-oss is a reasoning model: default effort burns ~4x the tokens and can hit the length limit
        # mid-thought (ragas LLMDidNotFinishException). Judging/extraction needs little reasoning.
        reasoning_effort="low",
        # 2048 ran out while extracting statements from long answers, which came back NaN
        max_completion_tokens=4096,
    )


@lru_cache(maxsize=1)
def get_judge_llm() -> BaseRagasLLM:
    # bypass_n: Groq rejects n>1, so multi-sample metrics (ResponseRelevancy) issue separate calls
    return LangchainLLMWrapper(get_judge_chat(), run_config=RUN_CONFIG, bypass_n=True)


@lru_cache(maxsize=1)
def get_judge_embeddings() -> BaseRagasEmbeddings:
    return LangchainEmbeddingsWrapper(LocalEmbeddings(), run_config=RUN_CONFIG)
