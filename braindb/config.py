import functools
import logging
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# LLM provider profiles. Flip the whole stack by setting LLM_PROFILE in .env.
# Every profile is an OpenAI-compatible Chat Completions endpoint (hosted
# gateways, vLLM, Ollama, LM Studio, llama.cpp): a model id, the env var
# holding its API key and the endpoint's base_url. Adding a new provider is a
# dict entry, no code change.
#
# `deepinfra` is the recommended default — fast, cheap, validated end-to-end
# in the wiki/maintainer/writer pipeline. The `vllm_*` profiles are for
# advanced / self-hosted / offline use and require a workstation GPU
# (typically reached over an SSH tunnel from the docker network).
_LLM_PROFILES: dict[str, dict[str, str]] = {
    "nim": {
        "model": "google/gemma-4-31b-it",
        "api_key_env": "NVIDIA_NIM_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
    },
    "deepinfra": {
        "model": "google/gemma-4-31B-it",
        "api_key_env": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
    },
    # Generic OpenAI-compatible /v1 endpoint (Ollama, LM Studio, copilot-api,
    # vLLM, ...). Unlike the vllm_* profiles, its base_url is NOT fixed here —
    # set OPENAI_BASE_URL in the env (see Settings.resolved_base_url). The
    # model has no default: set AGENT_MODEL to your served model, e.g.
    # AGENT_MODEL=llama3.2:3b.
    "openai_compatible": {
        "model": "",
        "api_key_env": "OPENAI_API_KEY",
    },
    "vllm_workstation": {
        "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8002/v1",
    },
    # Qwen 3.8 27B (NVFP4) on the workstation vLLM. This is the profile the wiki
    # pipeline is actually tuned against — the turn budget, request timeout and
    # handoff budget below all cite it. Select it and no AGENT_MODEL override is
    # needed.
    "vllm_workstation_qwen": {
        "model": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8012/v1",
    },
    "vllm_workstation_gemma": {
        "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8009/v1",
    },
    # Gemma 4 12B (QAT w4a16) on the workstation vLLM — a SMALLER, distinct model
    # from the 31B above, on its own port. Reached over the laptop/Pi -> workstation
    # tunnel to :8012.
    "vllm_workstation_gemma12b": {
        "model": "google/gemma-4-12B-it-qat-w4a16-ct",
        "api_key_env": "VLLM_API_KEY",
        "base_url": "http://host.docker.internal:8012/v1",
    },
}

# Hosted gateways: a real API key is always required (no "EMPTY"
# placeholder), and JEV, which needs a self-hosted vLLM, stays off.
_HOSTED_PROFILES = {"nim", "deepinfra"}

# The LiteLLM routing prefix an existing .env may still carry in AGENT_MODEL,
# per profile. One is dropped (with a warning) so old files keep working; a
# self-hosted model id of its own that starts with "openai/" (e.g. a served
# "openai/gpt-oss-20b") is then written with the prefix twice. Hosted ids such
# as DeepInfra's "openai/gpt-oss-120b" are never touched.
_LEGACY_PREFIXES = {"deepinfra": ("deepinfra/",), "nim": ("nvidia_nim/",)}
_SELF_HOSTED_LEGACY_PREFIXES = ("openai/", "hosted_vllm/")


@functools.cache
def _model_id(profile: str, model: str) -> str:
    for prefix in _LEGACY_PREFIXES.get(profile, _SELF_HOSTED_LEGACY_PREFIXES):
        if model.startswith(prefix):
            logging.getLogger(__name__).warning(
                "AGENT_MODEL %r: dropped the old LiteLLM prefix %r; the model id is %r",
                model, prefix, model[len(prefix):])
            return model[len(prefix):]
    return model


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

    # Agent (OpenAI Agents SDK — provider selected via llm_profile)
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
    # Per-LLM-call HTTP deadline (seconds), passed to the OpenAI client as
    # `timeout=`. Without it the client applies its own 600s default. 600s
    # is ample for a short recall, but a wiki writer regenerating a 50k-char
    # body on a local 27B can exceed it — the client then abandons a request
    # vLLM is still completing, so the work is computed and discarded. Set to 4800
    # (80 min) to cover a full 30-turn writer run at self-hosted latencies.
    # Hosted providers finish far inside this and are unaffected: the value
    # is a ceiling, not a delay.
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
    # on Qwen, fragmenting work across successors unnecessarily.
    # Raised 20000 -> 30000 after a long soak: on pages approaching
    # 50k chars the writer needs the successor path, and a budget set
    # too high simply never fires, leaving it to accumulate context
    # until it hits its turn limit. Keep this well inside the smallest
    # context window you deploy on — above it, handoff is silently
    # dead. Set to 0 to disable the handoff nudge entirely.
    agent_writer_handoff_token_budget: int = 30000
    agent_writer_handoff_max_depth: int = 3

    # Reasoning effort for the WIKI agents (maintainer / writer / subagent).
    # Blank = send nothing, i.e. the server-side default — no behaviour
    # change. Delivered as `chat_template_kwargs.reasoning_effort` in the
    # request body (see `agent._build` for why it cannot be sent as a plain
    # `reasoning_effort` param). SELF-HOSTED vLLM ONLY: a hosted provider may
    # reject the unknown body key, so leave it blank on those profiles.
    #
    # Why this exists: the Qwen3 chat template resolves
    # `reasoning_effort|default('xhigh')`, so sending nothing runs every
    # request at the MAXIMUM setting and injects a "think carefully,
    # validate assumptions, consider alternatives" instruction into each
    # system block. Measured on the bench box: ~2750 output tokens/turn
    # and ~99% of wall clock is generation, so trimming reasoning trims
    # latency almost linearly. It costs no cross-turn consistency either:
    # the SDK only replays reasoning for DeepSeek/Claude models, so on
    # Qwen every reasoning token is generated, paid for and then dropped
    # before the next turn.
    #
    # Valid values for that template: "low", "medium", "none". NOT
    # "minimal"/"high" — vLLM's Literal accepts them but the template's
    # own validator raises. Wiki-scoped on purpose: the general agent
    # (`get_agent`) is shared with the ingest watcher, whose extraction
    # runs are the most reasoning-dependent work in the stack.
    agent_wiki_reasoning_effort: str = ""

    # JEV — closed yes/no relevance judgements read from one token's
    # probability (services/jev.py). SELF-HOSTED vLLM ONLY: it needs vLLM's
    # token-id prompts, /tokenize and logprob_token_ids, and on any other
    # endpoint its first-use probe keeps it off. This switch is the recall
    # filter only: off by default, and with it off recall is unchanged. On, a
    # recall made while answering an /agent/query request that states a
    # `need` keeps only the items the model judges useful for that need. The
    # other JEV features below (wiki triage, wiki gate, reader, hops) have
    # switches of their own.
    jev_enabled: bool = False
    # Minimum P(useful) an item needs to be kept. Chosen on dev needs over a
    # news memory: every threshold from 0.01 to 0.4 kept the same essential
    # items (none scored inside that band), so 0.1 sits safely inside it.
    jev_strictness: float = 0.1
    # Wiki triage by JEV (services/wiki_triage.py): the maintainer's per-seed
    # decision (action + target wiki) taken as one-token picks instead of an
    # agent run; seeds it cannot decide confidently still go to the agent.
    # SELF-HOSTED vLLM ONLY, like the rest of JEV. Off by default, and off
    # under a custom profile that replaces the maintainer's prompt; a
    # profile's maintainer additions shape the agent's decisions only.
    jev_wiki_triage: bool = False
    # The wiki writer behind a JEV gate (services/wiki_gate.py): which facts
    # reach the writer (identity) and which sections of a long page it is
    # shown (placement) are one-token picks taken before it starts; the
    # writer then edits with only those, and no research tools. Attach and
    # create jobs only; anything the gate cannot settle runs the full writer
    # as before. SELF-HOSTED vLLM ONLY; off by default, and off under a
    # custom profile that replaces the writer's prompt.
    jev_wiki_gate: bool = False
    # The adaptive reader, nested in the agent's conversation
    # (services/reader.py): when an /agent/query request (or one of its
    # subagents) starts reading a body longer than one tool output, the body
    # is read once, piece by piece inside the agent's own conversation, each
    # piece kept or dropped by one-token picks, and the agent continues with
    # only what was kept. SELF-HOSTED vLLM ONLY; off by default, and with it
    # off get_entity is unchanged.
    jev_reader: bool = False
    # Hops (services/hops.py): after a recall_memory call in an /agent/query
    # request, the relation neighbours of what recall showed are offered one
    # at a time inside the agent's own conversation, each kept or dropped by
    # a one-token pick, and only the kept ones are followed to the next hop;
    # with the wiki gate on, the job members' neighbours join its candidates.
    # SELF-HOSTED vLLM ONLY; off by default, and with it off recall_memory
    # is unchanged.
    jev_hops: bool = False

    @property
    def resolved_agent_model(self) -> str:
        return _model_id(self.llm_profile,
                         self.agent_model or _LLM_PROFILES[self.llm_profile]["model"])

    @property
    def self_hosted(self) -> bool:
        """A configured base URL that is not one of the hosted gateways —
        normally a server you run (vLLM, Ollama, ...). JEV is only tried
        there (and probes the server before trusting it)."""
        return self.llm_profile not in _HOSTED_PROFILES and bool(self.resolved_base_url)

    @property
    def resolved_api_key(self) -> str:
        profile = _LLM_PROFILES[self.llm_profile]
        key = os.getenv(profile["api_key_env"], "")
        # Self-hosted servers (vLLM/Ollama) may run without auth, but the
        # OpenAI client still needs a non-empty key — supply a placeholder.
        # Covers the env-driven openai_compatible profile too, not just the
        # profiles with a fixed base_url.
        if not key and self.self_hosted:
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
