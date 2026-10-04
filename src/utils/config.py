"""Application configuration loaded from environment variables.

Every pipeline layer reads settings from a single frozen ``Config`` dataclass
built by :func:`load_config`. Nothing else in the codebase should call
``os.environ`` directly - this keeps configuration centralized and testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of all environment-driven settings."""

    llm_provider: str
    openai_api_key: str | None
    google_api_key: str | None
    groq_api_key: str | None
    whisper_model_size: str
    confidence_threshold: float
    low_confidence_halt_ratio: float
    db_path: str
    db_encryption_key: str | None
    max_retries_per_node: int
    llm_timeout_seconds: int
    langsmith_api_key: str | None
    langsmith_project: str | None
    langsmith_tracing: bool
    langfuse_enabled: bool
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str
    max_file_retention: int


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_config() -> Config:
    """Read all required and optional environment variables into a Config.

    Never raises on missing optional values - they simply come back as
    ``None`` or their documented default, so this is always safe to call at
    startup even before a ``.env`` file has been fully populated.
    """
    return Config(
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "tiny"),
        confidence_threshold=_get_float("CONFIDENCE_THRESHOLD", 0.55),
        low_confidence_halt_ratio=_get_float("LOW_CONFIDENCE_HALT_RATIO", 0.4),
        db_path=os.getenv("DB_PATH", "data/calls.db"),
        db_encryption_key=os.getenv("DB_ENCRYPTION_KEY") or None,
        max_retries_per_node=_get_int("MAX_RETRIES_PER_NODE", 3),
        llm_timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 60),
        langsmith_api_key=os.getenv("LANGCHAIN_API_KEY") or None,
        langsmith_project=os.getenv("LANGCHAIN_PROJECT") or None,
        langsmith_tracing=_get_bool("LANGCHAIN_TRACING_V2", False),
        langfuse_enabled=_get_bool("LANGFUSE_ENABLED", False),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or None,
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY") or None,
        langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        max_file_retention=_get_int("MAX_TEMP_FILE_RETENTION", 50),
    )
