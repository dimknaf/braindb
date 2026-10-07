"""
Typed agent output contract.

Convention (absolute): every agent/subagent finishes via the `final_answer`
trick, and its payload is ALWAYS one of these Pydantic models — never a loose
free string we scrape. `@function_tool` turns the model into a strict JSON
schema for the tool arguments, so the LLM is constrained to emit valid
structured output instead of free-running and truncating.

These mirror the style of `braindb/schemas/` (the REST layer); they reuse the
existing pydantic dependency — no new dependency, no new machinery.
"""
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# Coercion helpers — weak/quantised models often emit "" (empty string) for
# nullable fields instead of `null`, or `null` for empty-list fields instead
# of `[]`. The Pydantic schemas are nullable + defaulted at the type level;
# these `before` validators just accept the wrong-type variants gracefully
# so we don't reject a perfectly intended "skip" decision because the model
# sent `target_wiki_no=""` instead of `null`. The validation contract is
# unchanged — we still produce a properly-typed Pydantic instance.


# Top-level coercion — some providers (notably vLLM / Qwen) emit tool-call
# `arguments.payload` as a JSON-encoded STRING ("{\"action\": \"skip\", ...}")
# instead of a JSON object ({"action": "skip", ...}). This is technically
# OpenAI-spec-compliant (the outer `arguments` field IS defined as a string
# of JSON), but the SDK only unwraps once and then hands the inner value to
# Pydantic as-is — so when the inner value is itself a JSON string, Pydantic
# rejects it with "Input should be a valid dictionary".
#
# The `@model_validator(mode="before")` below catches this case: if the input
# is a string that parses as JSON to a dict, we use the parsed dict; if it
# parses to anything else (list / int / null), we let Pydantic raise its
# usual "valid dictionary" error so the LLM sees a clear correction. Dict
# inputs are passed through untouched — well-behaved providers (deepinfra,
# OpenAI-compatible servers) see exactly the same behaviour as today.
#
# This is the SAME pattern as the nullable-field coercion above, just at the
# whole-model level rather than per-field. The LLM-visible schema is
# unchanged; we don't advertise string-form acceptance to the model.

def _maybe_parse_json_string(v):
    """If `v` is a JSON-encoded string of an object, parse it. Otherwise
    pass through unchanged. Pydantic v2 calls @model_validator(mode='before')
    BEFORE field-level validation, so a returned dict goes through the rest
    of the validation pipeline (including the per-field coercers below)
    exactly as if the LLM had sent a dict in the first place."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return v  # let Pydantic raise its normal error
        # Defensive: occasionally Qwen-class quantised models emit the dict
        # double-escaped (the first parse yields a string of JSON, not a
        # dict). One more parse attempt unwraps that case. Safe — only fires
        # on a string result, only returns a value if it parses to a dict.
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except (json.JSONDecodeError, ValueError):
                return v
        # Only return the parsed value if it's a dict — anything else (list,
        # int, null) is not a valid Pydantic-model input; let Pydantic raise.
        if isinstance(parsed, dict):
            return parsed
    return v

def _coerce_empty_to_none(v):
    """Accept '', 'null', 'none', 'n/a' (any case, with/without whitespace)
    as equivalent to None for nullable fields."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in ("null", "none", "n/a"):
            return None
    return v


def _coerce_to_int_or_none(v):
    """For nullable-int fields: '' / 'null' / etc → None; numeric strings → int."""
    v = _coerce_empty_to_none(v)
    if v is None or isinstance(v, int):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return None  # last resort — don't fail the whole submission on a bad number


def _coerce_to_list(v):
    """For list fields: None / '' → []; everything else as-is for Pydantic to validate."""
    if v is None or v == "":
        return []
    return v


class AgentAnswer(BaseModel):
    """General recall/save answer (the public /agent/query endpoint).

    The endpoint is general-purpose (Claude Code, arbitrary recall/save), so
    the answer itself is necessarily natural language — but it is still
    delivered through the typed `final_answer` trick, never as loose
    top-level model output.
    """
    answer: str = Field(..., description="The full natural-language response to the caller.")

    # Safety net for providers (notably vLLM/Qwen) that emit the tool-call
    # arg as a JSON-encoded string instead of a JSON object. See the helper
    # docstring at the top of the file.
    @model_validator(mode="before")
    @classmethod
    def _accept_json_string(cls, v):
        return _maybe_parse_json_string(v)


class MaintainerDecision(BaseModel):
    """The wiki maintainer's per-orphan decision. Existing wikis are
    referenced by their CATALOG NUMBER (the numbered list at the end of the
    prompt), never by uuid — the harness maps number->id deterministically.

    Action-dependent fields: `target_wiki_no`, `proposed_name`, and
    `consolidate_nos` are only meaningful for one specific action each. For
    every other action you MUST send JSON `null` for the optional ones (not
    "", not 0, not "n/a") and an empty array `[]` for `consolidate_nos`.
    """
    action: Literal["attach", "create", "consolidate", "skip", "ambiguous"] = Field(
        ...,
        description=(
            "The decision for this orphan. Exactly one of: "
            "`attach` (link to an existing wiki by catalog number), "
            "`create` (mint a new wiki with a proposed name), "
            "`consolidate` (merge >=2 catalog-numbered wikis), "
            "`skip` (not worth a wiki — infrastructural / keyword-token), "
            "`ambiguous` (cannot disambiguate the real subject)."
        ),
    )
    target_wiki_no: int | None = Field(
        None,
        description=(
            "REQUIRED ONLY when action=`attach`: the integer CATALOG NUMBER "
            "of the existing wiki to attach the orphan to (1-indexed, taken "
            "from the numbered WIKIS list at the end of the prompt). "
            "For action in (`create`, `consolidate`, `skip`, `ambiguous`) "
            "this field MUST be JSON null. Do NOT use empty string \"\", 0, "
            "or 'n/a' — use literal null."
        ),
    )
    proposed_name: str | None = Field(
        None,
        description=(
            "REQUIRED ONLY when action=`create`: the canonical name for the "
            "new wiki (must appear in the evidence — never invent). "
            "For action in (`attach`, `consolidate`, `skip`, `ambiguous`) "
            "this field MUST be JSON null. Do NOT use empty string \"\"."
        ),
    )
    consolidate_nos: list[int] = Field(
        default_factory=list,
        description=(
            "REQUIRED ONLY when action=`consolidate`: an array of >=2 "
            "integer CATALOG NUMBERS naming the duplicate wikis to merge "
            "(from the numbered WIKIS list). "
            "For every other action this field MUST be an empty array [] "
            "(NOT null, NOT empty string)."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "ALWAYS REQUIRED. One to three sentences justifying the chosen "
            "action: which catalog wiki(s) you matched (or that the catalog "
            "has none), and why this action was the right one. This makes "
            "the decision auditable."
        ),
    )

    # Top-level coercion: accept JSON-string-of-dict (vLLM/Qwen quirk).
    @model_validator(mode="before")
    @classmethod
    def _accept_json_string(cls, v):
        return _maybe_parse_json_string(v)

    # Forgiving coercion — weak/quantised models often emit empty strings or
    # "null" strings instead of literal JSON null. Accept those as None
    # rather than rejecting the whole submission (the prompt and the
    # descriptions above ask for null; the validators are the safety net).
    @field_validator("target_wiki_no", mode="before")
    @classmethod
    def _coerce_target_wiki_no(cls, v):
        return _coerce_to_int_or_none(v)

    @field_validator("proposed_name", mode="before")
    @classmethod
    def _coerce_proposed_name(cls, v):
        return _coerce_empty_to_none(v)

    @field_validator("consolidate_nos", mode="before")
    @classmethod
    def _coerce_consolidate_nos(cls, v):
        return _coerce_to_list(v)


class MaintainerClusterItem(MaintainerDecision):
    """One per-seed decision inside a clustered triage. Identical to
    `MaintainerDecision` (same fields + the same forgiving coercion validators,
    inherited) plus the `entity_id` of the seed it applies to, so the harness
    maps each decision back to its orphan."""
    entity_id: str = Field(
        ...,
        description=(
            "The seed's entity_id, copied VERBATIM from the SEEDS list in the "
            "prompt. Identifies which orphan THIS decision is for."
        ),
    )


class MaintainerClusterDecision(BaseModel):
    """The maintainer's output for a triage cluster: one decision per seed. A
    singleton triage is just a cluster of one (a single-element list), so this
    is the maintainer's only output schema. The harness applies each item
    exactly as it applied a `MaintainerDecision` before."""
    decisions: list[MaintainerClusterItem] = Field(
        ...,
        description=(
            "Exactly one entry per seed in the SEEDS list — do not omit a seed "
            "and do not invent seeds. Each entry is a full decision (action + "
            "its action-specific fields + rationale) plus the seed's entity_id."
        ),
    )

    # Top-level coercion: accept JSON-string-of-dict (vLLM/Qwen quirk).
    @model_validator(mode="before")
    @classmethod
    def _accept_json_string(cls, v):
        return _maybe_parse_json_string(v)


class WikiWriteResult(BaseModel):
    """The wiki writer's full output. `body` is the complete markdown page —
    a typed field of the schema, exactly like any other field (not loose
    text, not delimiter-wrapped).

    `canonical_no` is only meaningful for `consolidate` mode. For
    `create` / `attach` you MUST send JSON null (not "", not 0).
    """
    mode: Literal["create", "attach", "consolidate"] = Field(
        ...,
        description=(
            "The write mode of THIS job (matches the mode the harness "
            "passed in the prompt): `create` (fresh wiki), `attach` "
            "(integrate new members into an existing wiki), `consolidate` "
            "(merge multiple duplicate wikis into a survivor)."
        ),
    )
    canonical_no: int | None = Field(
        None,
        description=(
            "REQUIRED ONLY when mode=`consolidate`: the integer NUMBER of "
            "the surviving wiki chosen from the numbered DUPLICATES list "
            "in the prompt (1-indexed, never a uuid). "
            "For mode in (`create`, `attach`) this field MUST be JSON null. "
            "Do NOT use empty string \"\", 0, or 'n/a'."
        ),
    )
    body: str = Field(
        "",
        description=(
            "The COMPLETE markdown wiki page — the full document. Include "
            "the meta header, summary, disambiguation, every section, all "
            "[[ref:UUID]] citations, and the references section. This is "
            "what becomes the wiki entity's content; it replaces the prior "
            "body wholesale (the prior version is auto-snapshotted).\n\n"
            "MAY be empty ONLY in `attach` mode AND only if you persisted "
            "your changes via the section-edit tools "
            "(`edit_wiki_section` / `delete_wiki_section`). In that case "
            "the router detects the wiki's revision moved during your run "
            "and skips the full-body write — your section edits are "
            "already the authoritative content. For `create` and "
            "`consolidate` modes this field MUST be non-empty."
        ),
    )

    # Top-level coercion: accept JSON-string-of-dict (vLLM/Qwen quirk).
    @model_validator(mode="before")
    @classmethod
    def _accept_json_string(cls, v):
        return _maybe_parse_json_string(v)

    @field_validator("canonical_no", mode="before")
    @classmethod
    def _coerce_canonical_no(cls, v):
        return _coerce_to_int_or_none(v)


class SubagentResult(BaseModel):
    """A delegated subagent's return (replaces the free-string subagent answer)."""
    result: str = Field(..., description="The distilled result of the delegated task.")

    # Top-level coercion: accept JSON-string-of-dict (vLLM/Qwen quirk).
    @model_validator(mode="before")
    @classmethod
    def _accept_json_string(cls, v):
        return _maybe_parse_json_string(v)
