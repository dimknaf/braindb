"""Deterministic fast paths for simple BrainDB agent requests."""
import re
from typing import Any

from braindb.agent.tools import _save_fact_impl, _save_rule_impl

_SAVE_RE = re.compile(r"^\s*Save:\s+(?P<content>.+?)\s*$", re.IGNORECASE | re.DOTALL)
_SAVE_RULE_RE = re.compile(r"^\s*Save as rule:\s+(?P<content>.+?)\s*$", re.IGNORECASE | re.DOTALL)
_MAX_FAST_PATH_CHARS = 2000


def _content_is_safe_for_fast_path(content: str) -> bool:
    return bool(content) and "?" not in content and len(content) <= _MAX_FAST_PATH_CHARS


def try_fast_path(query: str) -> dict[str, Any] | None:
    """Handle simple save requests without invoking the LLM agent loop."""
    rule_match = _SAVE_RULE_RE.match(query)
    if rule_match:
        content = rule_match.group("content").strip()
        if not _content_is_safe_for_fast_path(content):
            return None
        answer = _save_rule_impl(
            content=content,
            keywords=[],
            importance=0.8,
        )
        status = "fast_path_error" if answer.startswith("ERROR:") else "fast_path"
        return {"answer": answer, "max_turns": 0, "status": status}

    save_match = _SAVE_RE.match(query)
    if save_match:
        content = save_match.group("content").strip()
        if not _content_is_safe_for_fast_path(content):
            return None
        answer = _save_fact_impl(
            content=content,
            keywords=[],
            source="user-stated",
            certainty=0.9,
            importance=0.7,
            notes="Saved via agent fast path.",
        )
        status = "fast_path_error" if answer.startswith("ERROR:") else "fast_path"
        return {"answer": answer, "max_turns": 0, "status": status}

    return None
