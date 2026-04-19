import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# LLM provider profiles. Flip the whole stack by setting LLM_PROFILE in .env.
# Each profile is just a LiteLLM model prefix + the env var holding its API key.
# Adding a new provider is a dict entry, no code change.
_LLM_PROFILES: dict[str, dict[str, str]] = {
    "nim": {
        "model": "nvidia_nim/google/gemma-4-31b-it",
        "api_key_env": "NVIDIA_NIM_API_KEY",
    },
    "deepinfra": {
        "model": "deepinfra/google/gemma-4-31B-it",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
}


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
    agent_max_turns: int = 15
    agent_subagent_max_turns: int = 30
    agent_verbose: bool = False

    @property
    def resolved_agent_model(self) -> str:
        return self.agent_model or _LLM_PROFILES[self.llm_profile]["model"]

    @property
    def resolved_api_key(self) -> str:
        return os.getenv(_LLM_PROFILES[self.llm_profile]["api_key_env"], "")


settings = Settings()
