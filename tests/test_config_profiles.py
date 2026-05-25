import pytest

from braindb.config import Settings


pytestmark = pytest.mark.unit


def test_openai_compatible_profile_resolves_env_values(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai/gpt-5-mini")
    monkeypatch.setenv("AGENT_BASE_URL", "http://localhost:4141/v1")
    monkeypatch.setenv("AGENT_API_KEY", "test-key")

    settings = Settings(_env_file=None, llm_profile="openai_compatible")

    assert settings.resolved_agent_model == "openai/gpt-5-mini"
    assert settings.resolved_base_url == "http://localhost:4141/v1"
    assert settings.resolved_api_key == "test-key"


def test_openai_compatible_profile_allows_empty_key_for_local_endpoint(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai/llama3.2:3b")
    monkeypatch.setenv("AGENT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("AGENT_API_KEY", raising=False)

    settings = Settings(_env_file=None, llm_profile="openai_compatible")

    assert settings.resolved_agent_model == "openai/llama3.2:3b"
    assert settings.resolved_base_url == "http://localhost:11434/v1"
    assert settings.resolved_api_key == "EMPTY"


@pytest.mark.parametrize(
    ("profile", "expected_model", "expected_base_url"),
    [
        ("deepinfra", "deepinfra/google/gemma-4-31B-it", None),
        ("nim", "nvidia_nim/google/gemma-4-31b-it", None),
        ("vllm_workstation", "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit", "http://host.docker.internal:8002/v1"),
        ("vllm_workstation_qwen", "openai/cyankiwi/Qwen3.6-27B-AWQ-INT4", "http://host.docker.internal:8010/v1"),
        ("vllm_workstation_gemma", "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit", "http://host.docker.internal:8009/v1"),
    ],
)
def test_existing_profiles_keep_current_resolution(monkeypatch, profile, expected_model, expected_base_url):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    settings = Settings(_env_file=None, llm_profile=profile)

    assert settings.resolved_agent_model == expected_model
    assert settings.resolved_base_url == expected_base_url
    assert settings.resolved_api_key == ("EMPTY" if expected_base_url else "")
