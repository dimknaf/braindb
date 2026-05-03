import pytest

from braindb.config import Settings


pytestmark = pytest.mark.unit


def test_codex_profile_resolves_default_model():
    settings = Settings(_env_file=None, llm_profile="codex", agent_model="", openai_api_key="test-key")

    assert settings.resolved_agent_model == "openai/gpt-5.3-codex-spark"


def test_codex_profile_resolves_api_key_from_field():
    settings = Settings(_env_file=None, llm_profile="codex", openai_api_key="test-key")

    assert settings.resolved_api_key == "test-key"


def test_codex_profile_resolves_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    settings = Settings(_env_file=None, llm_profile="codex")

    assert settings.resolved_api_key == "env-key"


def test_agent_model_override_wins_for_codex_profile():
    settings = Settings(
        _env_file=None,
        llm_profile="codex",
        agent_model="openai/alternate-model",
        openai_api_key="test-key",
    )

    assert settings.resolved_agent_model == "openai/alternate-model"


def test_unknown_profile_error_lists_known_profiles():
    settings = Settings(_env_file=None, llm_profile="missing", agent_model="")

    with pytest.raises(ValueError, match="openai_compatible"):
        _ = settings.resolved_agent_model


def test_openai_compatible_profile_default_api_key():
    settings = Settings(
        _env_file=None,
        llm_profile="openai_compatible",
        agent_model="openai/gpt-5-mini",
        agent_base_url="http://localhost:4141/v1",
    )

    assert settings.resolved_agent_model == "openai/gpt-5-mini"
    assert settings.resolved_api_key == "ollama"
    assert settings.resolved_agent_base_url == "http://localhost:4141/v1"


def test_local_ollama_alias_matches_openai_compatible():
    from braindb.config import _LLM_PROFILES

    assert _LLM_PROFILES["local_ollama"] is _LLM_PROFILES["openai_compatible"]
    settings = Settings(
        _env_file=None,
        llm_profile="local_ollama",
        agent_model="openai/llama3.2:3b",
    )

    assert settings.resolved_agent_model == "openai/llama3.2:3b"
    assert settings.resolved_api_key == "ollama"
