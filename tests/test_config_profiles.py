"""
Pure-unit coverage for LLM provider profile resolution in braindb.config.

No live stack: these construct Settings() directly with _env_file=None and
monkeypatched env, so they run with `pytest` even when the test stack is down
(they request no `api` fixture). The focus is the `openai_compatible` profile
and — critically — that its env-driven base URL does NOT leak into any other
profile's resolution (the scoping guarantee from PR #7's review).
"""
import pytest

from braindb.config import Settings


def test_openai_compatible_resolves_env_values(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai/gpt-5-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:4141/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    s = Settings(_env_file=None, llm_profile="openai_compatible")

    assert s.resolved_agent_model == "gpt-5-mini"  # legacy "openai/" prefix accepted
    assert s.resolved_base_url == "http://localhost:4141/v1"
    assert s.resolved_api_key == "test-key"


def test_openai_compatible_empty_key_falls_back_to_placeholder(monkeypatch):
    """Local endpoints (Ollama) run without auth — the OpenAI client still
    needs a non-empty key, so the resolver supplies the 'EMPTY' placeholder."""
    monkeypatch.setenv("AGENT_MODEL", "openai/llama3.2:3b")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    s = Settings(_env_file=None, llm_profile="openai_compatible")

    assert s.resolved_base_url == "http://localhost:11434/v1"
    assert s.resolved_api_key == "EMPTY"


def test_openai_compatible_without_base_url_is_none(monkeypatch):
    """No OPENAI_BASE_URL → no base URL, and (no key) → real empty key, not
    the placeholder (the placeholder is only for self-hosted/base_url paths)."""
    monkeypatch.setenv("AGENT_MODEL", "openai/llama3.2:3b")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    s = Settings(_env_file=None, llm_profile="openai_compatible")

    assert s.resolved_base_url is None
    assert s.resolved_api_key == ""


def test_openai_base_url_does_not_leak_into_other_profiles(monkeypatch):
    """The scoping guarantee: OPENAI_BASE_URL is honored ONLY by
    openai_compatible. Every other profile keeps its table base_url. A global
    override would be a regression."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://evil-override:9999/v1")

    vllm = Settings(_env_file=None, llm_profile="vllm_workstation")
    assert vllm.resolved_base_url == "http://host.docker.internal:8002/v1"

    deepinfra = Settings(_env_file=None, llm_profile="deepinfra")
    assert deepinfra.resolved_base_url == "https://api.deepinfra.com/v1/openai"

    nim = Settings(_env_file=None, llm_profile="nim")
    assert nim.resolved_base_url == "https://integrate.api.nvidia.com/v1"


@pytest.mark.parametrize(
    ("profile", "expected_model", "expected_base_url"),
    [
        ("deepinfra", "deepseek-ai/DeepSeek-V4.1-Flash", "https://api.deepinfra.com/v1/openai"),
        ("nim", "google/gemma-4-31b-it", "https://integrate.api.nvidia.com/v1"),
        ("vllm_workstation", "cyankiwi/gemma-4-31B-it-AWQ-4bit",
         "http://host.docker.internal:8002/v1"),
    ],
)
def test_existing_profiles_unchanged(monkeypatch, profile, expected_model, expected_base_url):
    """Regression: the established profiles keep their model and endpoint."""
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    s = Settings(_env_file=None, llm_profile=profile)

    assert s.resolved_agent_model == expected_model
    assert s.resolved_base_url == expected_base_url


@pytest.mark.parametrize(
    ("profile", "agent_model", "expected"),
    [
        ("deepinfra", "deepinfra/some/other-model", "some/other-model"),   # old LiteLLM form
        ("deepinfra", "some/other-model", "some/other-model"),
        ("deepinfra", "openai/gpt-oss-120b", "openai/gpt-oss-120b"),       # a real hosted id, untouched
        ("nim", "nvidia_nim/google/gemma-4-31b-it", "google/gemma-4-31b-it"),
        ("vllm_workstation", "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit", "cyankiwi/gemma-4-31B-it-AWQ-4bit"),
        ("vllm_workstation", "openai/openai/gpt-oss-20b", "openai/gpt-oss-20b"),  # served id starting openai/
    ],
)
def test_agent_model_override_still_wins(monkeypatch, profile, agent_model, expected):
    """AGENT_MODEL overrides the profile default for any profile. The old
    LiteLLM routing prefix of THAT profile is dropped, so existing .env files
    keep working; a model id's own organisation part is never touched."""
    monkeypatch.setenv("AGENT_MODEL", agent_model)
    assert Settings(_env_file=None, llm_profile=profile).resolved_agent_model == expected


def test_hosted_profiles_never_get_the_self_hosted_placeholder(monkeypatch):
    """Hosted gateways need a real key: without one the request must fail at
    the provider, not travel with the 'EMPTY' placeholder meant for local
    servers. They are also not self-hosted, so JEV stays off."""
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    s = Settings(_env_file=None, llm_profile="deepinfra")
    assert s.resolved_api_key == "" and not s.self_hosted
    assert Settings(_env_file=None, llm_profile="vllm_workstation").self_hosted
