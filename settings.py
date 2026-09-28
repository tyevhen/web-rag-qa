from functools import lru_cache

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

    # LangSmith tracing — all optional; tracing is disabled when unset
    langchain_tracing_v2: bool = False
    langchain_api_key: SecretStr | None = None
    langchain_project: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
