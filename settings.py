from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Required by the API server (uvicorn)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: SecretStr
    database_url: SecretStr
    groq_model: str
    system_prompt: str
    ingest_api_key: SecretStr
    redis_url: str = "redis://localhost:6379"
    allowed_origins: str = "*"
    # OpenAI-compatible endpoint for the app LLM; point at https://router.huggingface.co/v1 to use HF
    llm_base_url: str = "https://api.groq.com/openai/v1"
    hf_token: SecretStr | None = None
    # Seconds per LLM attempt; a local 7B model on a laptop needs ~180 for the reasoner prompt
    llm_timeout: float = 30

    # The encoder search uses (agent/embed.py ENCODERS). Switch only once every chunk has its vectors (backfill).
    embedder: str = "BAAI/bge-small-en-v1.5"
    # Also written at ingestion, to prepare a switch or compare encoders; JSON list in env
    shadow_embedders: list[str] = []

    # Evaluation judge — should be a different (ideally stronger) model than groq_model
    eval_judge_provider: Literal["groq", "hf"] = "groq"
    eval_judge_model: str = "openai/gpt-oss-20b"
    eval_sample_size: int = 20

    # LangSmith tracing — all optional; tracing is disabled when unset
    langchain_tracing_v2: bool = False
    langchain_api_key: SecretStr | None = None
    langchain_project: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
