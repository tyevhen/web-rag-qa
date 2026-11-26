from functools import lru_cache
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from ragas import RunConfig
from ragas.embeddings import BaseRagasEmbeddings, LangchainEmbeddingsWrapper
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper

from agent.embed import ENCODERS, embed_documents
from settings import get_settings

# Sequential + patient retries: free hosted tiers allow a few thousand tokens/min, and 429s surface as NaN scores
RUN_CONFIG = RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=180)


# Multilingual, so Estonian answers are scored as fairly as English ones; scores before this switch are not comparable
_JUDGE_ENCODER = ENCODERS["intfloat/multilingual-e5-small"]


class LocalEmbeddings(Embeddings):
    """LangChain adapter over the app's own e5 embedder — local, no API cost."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Pinned, not Settings.embedder: comparing search encoders must not change how the judge scores
        return embed_documents(texts, _JUDGE_ENCODER)

    def embed_query(self, text: str) -> list[float]:
        # Same prefix as documents: answer_relevancy compares question to question, so both sides must embed alike
        return embed_documents([text], _JUDGE_ENCODER)[0]


def judge_model_name() -> str:
    s = get_settings()
    host = f"{urlparse(s.eval_judge_base_url or '').hostname or 'unset'}:{s.eval_judge_model}"
    return f"{host}@{s.eval_judge_provider}" if s.eval_judge_provider else host


@lru_cache(maxsize=1)
def get_judge_chat() -> ChatOpenAI:
    s = get_settings()
    if s.eval_judge_base_url is None or s.eval_judge_api_key is None:
        raise ValueError("The judge needs EVAL_JUDGE_BASE_URL and EVAL_JUDGE_API_KEY")
    return ChatOpenAI(
        model=s.eval_judge_model,
        api_key=s.eval_judge_api_key,
        base_url=s.eval_judge_base_url,
        # No bursts: at ~1.5k tokens per call, EVAL_JUDGE_RPM keeps a run under the endpoint's tokens/min limit
        rate_limiter=InMemoryRateLimiter(requests_per_second=s.eval_judge_rpm / 60, max_bucket_size=1),
        temperature=0,
        # gpt-oss is a reasoning model: default effort burns ~4x the tokens and can hit the length limit
        # mid-thought (ragas LLMDidNotFinishException). Judging/extraction needs little reasoning.
        reasoning_effort="low",
        # 2048 ran out while extracting statements from long answers, which came back NaN
        max_completion_tokens=4096,
        # OpenRouter routing: fail rather than fall back to another host
        extra_body={"provider": {"order": [s.eval_judge_provider], "allow_fallbacks": False}}
        if s.eval_judge_provider else None,
    )


@lru_cache(maxsize=1)
def get_judge_llm() -> BaseRagasLLM:
    # bypass_n: some providers reject n>1, so multi-sample metrics (ResponseRelevancy) issue separate calls
    return LangchainLLMWrapper(get_judge_chat(), run_config=RUN_CONFIG, bypass_n=True)


@lru_cache(maxsize=1)
def get_judge_embeddings() -> BaseRagasEmbeddings:
    return LangchainEmbeddingsWrapper(LocalEmbeddings(), run_config=RUN_CONFIG)
