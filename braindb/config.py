import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# LLM provider profiles. Flip the whole stack by setting LLM_PROFILE in .env.
# Each profile is a LiteLLM model prefix + the env var holding its API key,
# plus an optional base_url for self-hosted OpenAI-compatible servers (vLLM,
# Ollama, llama.cpp). Adding a new provider is a dict entry, no code change.
_LLM_PROFILES: dict[str, dict[str, str]] = {
    "codex": {
        "model": "openai/gpt-5.3-codex-spark",
        "api_key_env": "OPENAI_API_KEY",
    },
    "nim": {
        "model": "nvidia_nim/google/gemma-4-31b-it",
        "api_key_env": "NVIDIA_NIM_API_KEY",
    },
    "deepinfra": {
        "model": "deepinfra/google/gemma-4-31B-it",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
    "openai_compatible": {
        "model": "",
        "api_key_env": "AGENT_API_KEY",
        "default_api_key": "ollama",
    },
    "vllm_workstation": {
        "model": "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8002/v1",
    },
}
_LLM_PROFILES["local_ollama"] = _LLM_PROFILES["openai_compatible"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://braindb:braindb@localhost:5432/braindb"
    api_port: int = 8000

    # Temporal decay rates per entity type (per day)
    decay_rate_thought: float = 0.005
    decay_rate_fact: float = 0.001
    decay_rate_source: float = 0.002
    decay_rate_datasource: float = 0.001
    decay_rate_rule: float = 0.0

    # Graph traversal
    max_graph_depth: int = 3
    min_relevance_threshold: float = 0.05
    level_decay: list[float] = [1.0, 0.6, 0.3]

    # Scoring
    missing_signal_penalty: float = 0.5   # multiplier when only text OR only embedding matches (0-1)

    # Always-on rules cap
    max_always_on_rules: int = 10

    # Agent (LiteLLM — provider selected via llm_profile)
    llm_profile: str = "deepinfra"
    agent_model: str = ""          # blank = use profile's default model
    agent_base_url: str = ""       # OpenAI-compatible base URL, e.g. http://host:11434/v1
    agent_api_key: str = ""        # optional generic key for OpenAI-compatible endpoints
    openai_api_key: str = ""
    deepinfra_api_key: str = ""
    nvidia_nim_api_key: str = ""
    agent_max_turns: int = 15
    agent_subagent_max_turns: int = 30
    agent_verbose: bool = False

    @property
    def _active_llm_profile(self) -> dict[str, str]:
        try:
            return _LLM_PROFILES[self.llm_profile]
        except KeyError as exc:
            known = ", ".join(sorted(_LLM_PROFILES))
            raise ValueError(f"Unknown LLM_PROFILE={self.llm_profile!r}. Expected one of: {known}") from exc

    def _env_setting(self, env_name: str) -> str:
        field_name = env_name.lower()
        return getattr(self, field_name, "") or os.getenv(env_name, "")

    @property
    def resolved_agent_model(self) -> str:
        model = self.agent_model or self._active_llm_profile["model"]
        if not model:
            raise ValueError(
                f"AGENT_MODEL must be set for LLM_PROFILE={self.llm_profile!r}; "
                "for OpenAI-compatible endpoints use AGENT_MODEL=openai/<model-id> "
                "(for example, openai/gpt-5-mini for copilot-api)."
            )
        return model

    @property
    def resolved_api_key(self) -> str:
        profile = self._active_llm_profile
        key = self._env_setting(profile["api_key_env"])
        if key:
            return key
        if "default_api_key" in profile:
            return profile["default_api_key"]
        if self.resolved_base_url:
            return "EMPTY"
        return ""

    @property
    def resolved_base_url(self) -> str | None:
        return self.agent_base_url or self._active_llm_profile.get("base_url")

    @property
    def resolved_agent_base_url(self) -> str | None:
        return self.resolved_base_url


settings = Settings()
