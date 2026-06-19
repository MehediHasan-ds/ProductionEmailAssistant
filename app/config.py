"""Application settings. One source of truth, validated at boot, loaded from .env."""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Production Email Assistant"
    default_provider: str = "openrouter"

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-oss-120b"
    openrouter_api_key: SecretStr

    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_api_key: SecretStr

    max_attempts: int = 3
    pass_threshold: float = 80.0  # TODO: tune after eval

    llm_timeout: float = 60.0
    llm_retry_attempts: int = 3
    llm_retry_max_wait: int = 10
    generation_timeout: float = 20.0

    # No machine-specific default; this must be provided via the environment.
    embedder_model_dir: str
    embedder_onnx_file: str = "model_quantized.onnx"
    embedder_max_length: int = 512
    embedder_prefix: str = "Document: "


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


PROVIDERS = ("openrouter", "gemini")
