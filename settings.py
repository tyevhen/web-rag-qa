import os
from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Required by the API server (uvicorn)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: SecretStr
    # The app LLM: a model at llm_base_url, e.g. the local Ollama model web-rag-llm
    llm_model: str
    system_prompt: str
    ingest_api_key: SecretStr
    redis_url: str = "redis://localhost:6379"
    allowed_origins: str = "*"
    # Any OpenAI-compatible endpoint for the app LLM; local Ollama by default, which needs no key
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr | None = None
    # Seconds per LLM attempt; a local 7B model on a laptop needs ~180 for the reasoner prompt
    llm_timeout: float = 180
    # Seconds for a whole /ask run, all LLM attempts included
    request_timeout: float = 600
    # Where the dashboard's Chat page reaches the API
    api_url: str = "http://localhost:8000"

    # The encoder search uses (agent/embed.py ENCODERS). Switch only once every chunk has its vectors (backfill).
    embedder: str = "BAAI/bge-small-en-v1.5"
    # Also written at ingestion, to prepare a switch or compare encoders; JSON list in env
    shadow_embedders: list[str] = []

    # Evaluation judge: any OpenAI-compatible endpoint, a different (ideally stronger) model than llm_model.
    # Only the eval runner, the nightly cron and the Chat Score button need it.
    eval_judge_base_url: str | None = None
    eval_judge_api_key: SecretStr | None = None
    eval_judge_model: str = "openai/gpt-oss-20b"
    eval_sample_size: int = 20

    # LangSmith tracing — all optional; tracing is disabled when unset
    langchain_tracing_v2: bool = False
    langchain_api_key: SecretStr | None = None
    langchain_project: str | None = None

    def export_langsmith_env(self) -> None:
        """LangChain reads tracing config from os.environ only, so copy it there; a no-op when tracing is off."""
        if not self.langchain_tracing_v2 or self.langchain_api_key is None:
            return
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key.get_secret_value()
        if self.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
