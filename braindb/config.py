import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# LLM provider profiles. Flip the whole stack by setting LLM_PROFILE in .env.
# Each profile is a LiteLLM model prefix + the env var holding its API key,
# plus an optional base_url for self-hosted OpenAI-compatible servers (vLLM,
# Ollama, llama.cpp). Adding a new provider is a dict entry, no code change.
#
# `deepinfra` is the recommended default — fast, cheap, validated end-to-end
# in the wiki/maintainer/writer pipeline. The `vllm_*` profiles are for
# advanced / self-hosted / offline use and require a workstation GPU
# (typically reached over an SSH tunnel from the docker network).
_LLM_PROFILES: dict[str, dict[str, str]] = {
    "nim": {
        "model": "nvidia_nim/google/gemma-4-31b-it",
        "api_key_env": "NVIDIA_NIM_API_KEY",
    },
    "deepinfra": {
        "model": "deepinfra/google/gemma-4-31B-it",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
    # Generic OpenAI-compatible /v1 endpoint (Ollama, LM Studio, copilot-api,
    # vLLM, ...). Unlike the vllm_* profiles, its base_url is NOT fixed here —
    # set OPENAI_BASE_URL in the env (see Settings.resolved_base_url). The
    # model has no default: set AGENT_MODEL to your served model with the
    # openai/ prefix, e.g. AGENT_MODEL=openai/llama3.2:3b.
    "openai_compatible": {
        "model": "",
        "api_key_env": "OPENAI_API_KEY",
    },
    "vllm_workstation": {
        "model": "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8002/v1",
    },
    "vllm_workstation_qwen": {
        "model": "openai/cyankiwi/Qwen3.6-27B-AWQ-INT4",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8010/v1",
    },
    "vllm_workstation_gemma": {
        "model": "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8009/v1",
    },
    # Gemma 4 12B (QAT w4a16) on the workstation vLLM — a SMALLER, distinct model
    # from the 31B above, on its own port. Reached over the laptop/Pi -> workstation
    # tunnel to :8012.
    "vllm_workstation_gemma12b": {
        "model": "openai/google/gemma-4-12B-it-qat-w4a16-ct",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8012/v1",
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
    decay_rate_wiki: float = 0.0   # synthesised pages should not fade

    # Graph traversal
    max_graph_depth: int = 3
    min_relevance_threshold: float = 0.05
    level_decay: list[float] = [1.0, 0.6, 0.3]

    # Scoring
    missing_signal_penalty: float = 0.5   # multiplier when only text OR only embedding matches (0-1)

    # Content witness — the `w` in `seed = keyword_score + w * c`, where `c`
    # (0-1) measures whether the entity BODY literally talks about the query
    # (stemmed full-text + per-word trigram fuzz, both index-backed). The
    # term is additive and boost-only: it can reinforce or introduce an
    # entity, never lower one. 0.0 disables the signal entirely (pure
    # keyword-mediated behavior).
    content_weight: float = 0.3
    # Size damper for the content witness: body-derived evidence is scaled by
    # 1 / (1 + log10(1 + chars/pivot)) — a big document must match BETTER
    # than a short fact to earn the same c. Bigger pivot = gentler damping.
    content_size_pivot: int = 2000

    # Scoring-pool caps. These bound the CANDIDATE pool that feeds ranking
    # (pure SQL/vector work — cheap, runs once per query). They are NOT the
    # LLM-visible cap; the caller's `max_results` truncates the FINAL sorted
    # items list. Keeping these wide is essential: a narrow single-word
    # keyword (e.g. "Petros") embedded against a multi-word sentence query
    # may not place in the top 30 most-similar keywords even when it's the
    # exact match — without enough headroom, nothing tagged with that
    # keyword enters the scoring pool at all.
    scoring_pool_keyword_neighbors: int = 500   # top-K keyword embeddings to consider
    scoring_pool_fuzzy: int = 500               # top-K fuzzy/full-text candidates to consider

    # Two-level diversity quota on recall output.
    #
    # Level 1 — per-search-term: each query string in `queries[]` gets
    # `per_query_share / num_queries` of `max_results` reserved for its
    # OWN top-ranked entities. Forces multi-angle representation: if
    # the agent issues [narrow_keyword, broader_phrase, third_angle],
    # all three angles surface in the result, regardless of which one
    # has the highest absolute scores. Set per_query_share=0 to disable.
    #
    # Level 2 — per-keyword (dominant matched keyword): walks the
    # remaining (open) slots in `final_rank` order and gives each new
    # dominant keyword a halving slot allowance (50% / 25% / 12.5% ...
    # of max_results, floor 1). Stops one popular keyword (e.g.
    # `user-profile`) from monopolising the open portion.
    #
    # The two levels share ONE counter dict — L1 reservations decrement
    # the same per-keyword allowance L2 walks against. So a popular
    # keyword cannot double-spend across the two layers.
    per_query_share: float = 0.5
    keyword_quota_halving: float = 0.5

    # How many entities the LLM-facing recall (`recall_memory` tool /
    # `/memory/context` API) returns by default. Wider default = the LLM
    # sees more candidates per call (more diverse, more discoverable),
    # at the cost of more prompt tokens. Tune in code, not via .env, so
    # all deployments share one measure.
    recall_default_max_results: int = 30

    # Always-on rules cap
    max_always_on_rules: int = 10

    # Agent (LiteLLM — provider selected via llm_profile)
    llm_profile: str = "deepinfra"
    agent_model: str = ""          # blank = use profile's default model
    # Base URL for the `openai_compatible` profile ONLY (e.g. Ollama on
    # http://host.docker.internal:11434/v1). Ignored by every other profile —
    # their base_url stays fixed in _LLM_PROFILES. See resolved_base_url.
    openai_base_url: str = ""
    # Bumped 15 → 20 after live observation on Qwen 27B AWQ-INT4 (vLLM):
    # deep-research-style runs commonly need >15 tool turns to land
    # `final_answer`. 20 gives breathing room; finishes-fast providers
    # (deepinfra/Gemma) are unaffected because they don't get close. Lower
    # than ~15 will regress Qwen behaviour. Callers that need a different
    # value (wiki maintainer/writer pass 30; ingest watcher passes 30/40)
    # still do so explicitly via `max_turns=` overrides.
    agent_max_turns: int = 20
    agent_subagent_max_turns: int = 30
    agent_verbose: bool = False
    # Per-LLM-call HTTP deadline (seconds), passed through to LiteLLM as
    # `timeout=`. Without it LiteLLM applies its own 600s fallback: its
    # `request_timeout` default is the sentinel 6000, and chat `completion()`
    # maps that sentinel down to `COMPLETION_HTTP_FALLBACK_SECONDS` (600)
    # whenever the caller sets no explicit timeout. 600s is ample for a
    # short recall, but a wiki writer regenerating a 50k-char body on a
    # local 27B can exceed it — the client then abandons a request vLLM is
    # still completing, so the work is computed and discarded. Set to 4800
    # (80 min) to cover a full 30-turn writer run at self-hosted latencies.
    # Hosted providers finish far inside this and are unaffected: the value
    # is a ceiling, not a delay. NOTE: setting the `REQUEST_TIMEOUT` env var
    # to exactly 6000 is a no-op (it IS the sentinel); this setting avoids
    # that trap by passing the value per-call instead.
    agent_request_timeout: int = 4800

    # Runtime "start wrapping up, you have N turns left" nudge (Layer 3 of
    # Stage C). When ≤ this many LLM-call turns remain before `max_turns`
    # is exhausted, `CountdownHooks` injects ONE synthetic user message
    # into the running conversation reminding the model to start
    # concluding research and call `final_answer`. The message tone is
    # context-aware: soft "start wrapping up" when `max_turns` is generous
    # (> 5), hard "call final_answer NOW" when the budget is tight (≤ 5,
    # which naturally covers the Layer 4 retry path with `max_turns=3`).
    # One nudge per run, never spammed. Set to 0 to disable entirely.
    # Bumped 5 → 8 so the nudge fires earlier and the model has room to
    # wrap up cleanly instead of slamming into the wall at the last turn.
    agent_countdown_threshold: int = 8

    # Retry-with-correction when a run ends without `final_answer` (Layer 4
    # of Stage C). If the model emits prose instead of calling the typed
    # termination tool, instead of raising immediately we append a synthetic
    # user-role correction message ("you ended without final_answer, call
    # it now") to the existing conversation (via `RunResult.to_input_list()`)
    # and re-invoke `Runner.run` ONCE with a small budget. If the retry
    # produces `final_answer` -> return the typed payload (HTTP 200). If the
    # retry ALSO fails -> raise `RuntimeError` (strict; no silent success
    # on a model that refuses the contract even after correction).
    # Bounded by `agent_retry_max_turns`; opt-out via setting to False.
    agent_retry_on_missing_final: bool = True
    agent_retry_max_turns: int = 3

    # Writer-only context-handoff threshold. When the cheap token estimate
    # of the writer's running conversation crosses this absolute number,
    # `CountdownHooks` injects ONE synthetic user message asking the model
    # to call `handoff_to_successor` with a structured brief (progress +
    # remaining work). The writer's run wrapper in `routers/wiki.py` then
    # spawns a successor agent (same prompt + tools, fresh context) seeded
    # with that brief. Bounded by `agent_writer_handoff_max_depth` so a
    # misbehaving model cannot thrash forever.
    #
    # Why a single absolute-token knob rather than a per-profile pct:
    # avoids per-profile bookkeeping. Tuned for the main production
    # target (Qwen 27B at max_model_len=40960, so 20000 ≈ 49% — fires
    # only when context is genuinely close to half-full). On the
    # hosted-Gemma 32K path 20000 is also safe (~63%). On the local
    # Gemma 13K path the budget is above the window so handoff never
    # fires — that's fine because the small-context path fails at
    # initial prompt construction long before the handoff can help.
    # Default was 9000 during the Phase-3 dry run; observation showed
    # that fired the handoff on routine consolidates that fit inline
    # on Qwen, fragmenting work across successors unnecessarily. Set
    # to 0 to disable the handoff nudge entirely.
    agent_writer_handoff_token_budget: int = 20000
    agent_writer_handoff_max_depth: int = 3

    @property
    def resolved_agent_model(self) -> str:
        return self.agent_model or _LLM_PROFILES[self.llm_profile]["model"]

    @property
    def resolved_api_key(self) -> str:
        profile = _LLM_PROFILES[self.llm_profile]
        key = os.getenv(profile["api_key_env"], "")
        # Self-hosted profiles (vLLM/Ollama) may run without auth, but the
        # OpenAI client still needs a non-empty key — supply a placeholder.
        # Uses resolved_base_url so the env-driven openai_compatible profile
        # is covered too, not just the profiles with a fixed base_url.
        if not key and self.resolved_base_url:
            return "EMPTY"
        return key

    @property
    def resolved_base_url(self) -> str | None:
        # Only the generic openai_compatible profile takes its base_url from
        # the environment (OPENAI_BASE_URL). Every other profile's base_url
        # stays fixed in the _LLM_PROFILES table — no silent global override.
        if self.llm_profile == "openai_compatible":
            return self.openai_base_url or None
        return _LLM_PROFILES[self.llm_profile].get("base_url")


settings = Settings()
