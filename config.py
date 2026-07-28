"""Centralized application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when application settings are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(repr=False)
    chat_model: str
    embedding_model: str
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 100
    retrieval_top_k: int = 4
    answer_temperature: float = 0.2
    summary_temperature: float = 0.3
    recent_message_limit: int = 6
    embedding_batch_size: int = 100
    request_timeout_seconds: float = 60.0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc


def get_settings(require_api_key: bool = True) -> Settings:
    """Load and validate settings, with environment variables taking precedence."""

    load_dotenv()
    settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.6-terra").strip(),
        embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ).strip(),
        chunk_size_tokens=_env_int("CHUNK_SIZE_TOKENS", 600),
        chunk_overlap_tokens=_env_int("CHUNK_OVERLAP_TOKENS", 100),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 4),
        answer_temperature=_env_float("ANSWER_TEMPERATURE", 0.2),
        summary_temperature=_env_float("SUMMARY_TEMPERATURE", 0.3),
        recent_message_limit=_env_int("RECENT_MESSAGE_LIMIT", 6),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 100),
        request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 60.0),
    )

    if require_api_key and not settings.openai_api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is missing. Add it to a local .env file or environment."
        )
    if not settings.chat_model:
        raise ConfigurationError("OPENAI_CHAT_MODEL cannot be empty.")
    if not settings.embedding_model:
        raise ConfigurationError("OPENAI_EMBEDDING_MODEL cannot be empty.")
    if settings.chunk_size_tokens <= 0:
        raise ConfigurationError("CHUNK_SIZE_TOKENS must be positive.")
    if not 0 <= settings.chunk_overlap_tokens < settings.chunk_size_tokens:
        raise ConfigurationError(
            "CHUNK_OVERLAP_TOKENS must be at least 0 and smaller than "
            "CHUNK_SIZE_TOKENS."
        )
    if settings.retrieval_top_k <= 0:
        raise ConfigurationError("RETRIEVAL_TOP_K must be positive.")
    if settings.embedding_batch_size <= 0:
        raise ConfigurationError("EMBEDDING_BATCH_SIZE must be positive.")
    if settings.recent_message_limit < 0:
        raise ConfigurationError("RECENT_MESSAGE_LIMIT cannot be negative.")
    if settings.request_timeout_seconds <= 0:
        raise ConfigurationError("REQUEST_TIMEOUT_SECONDS must be positive.")
    for name, value in (
        ("ANSWER_TEMPERATURE", settings.answer_temperature),
        ("SUMMARY_TEMPERATURE", settings.summary_temperature),
    ):
        if not 0.0 <= value <= 2.0:
            raise ConfigurationError(f"{name} must be between 0 and 2.")
    return settings
