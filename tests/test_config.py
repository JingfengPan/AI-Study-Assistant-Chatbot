from __future__ import annotations

import pytest

import config
from config import ConfigurationError, get_settings


def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda: False)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_CHAT_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "CHUNK_SIZE_TOKENS",
        "CHUNK_OVERLAP_TOKENS",
        "RETRIEVAL_TOP_K",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_without_api_key(monkeypatch, tmp_path):
    _clear_settings(monkeypatch)
    monkeypatch.chdir(tmp_path)
    settings = get_settings(require_api_key=False)
    assert settings.chat_model == "gpt-5.6-terra"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chunk_size_tokens == 600
    assert "openai_api_key" not in repr(settings)


def test_environment_override(monkeypatch, tmp_path):
    _clear_settings(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "test-chat")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "321")
    settings = get_settings()
    assert settings.openai_api_key == "test-key"
    assert settings.chat_model == "test-chat"
    assert settings.chunk_size_tokens == 321


def test_missing_required_key(monkeypatch, tmp_path):
    _clear_settings(monkeypatch)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        get_settings()


def test_invalid_overlap(monkeypatch, tmp_path):
    _clear_settings(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "100")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "100")
    with pytest.raises(ConfigurationError, match="smaller"):
        get_settings(require_api_key=False)


def test_invalid_top_k(monkeypatch, tmp_path):
    _clear_settings(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RETRIEVAL_TOP_K", "0")
    with pytest.raises(ConfigurationError, match="positive"):
        get_settings(require_api_key=False)
