"""
Optional custom-profile prompt composition. ZERO effect unless CUSTOM_PROFILE is set.

    CUSTOM_PROFILE=cityfalcon_news                # one profile
    CUSTOM_PROFILE=company_base,cityfalcon_news   # several, composed in order
    CUSTOM_PROFILE_DIR=custom-profiles            # optional; default shown

A profile is a folder under CUSTOM_PROFILE_DIR holding prompt fragments named
`<target>.add.md` (appended to the base prompt) and/or `<target>.replace.md`
(swaps the base template entirely). `target` is the prompt key a call site asks
for, e.g. "wiki_maintainer" or "wiki_writer".

Two composition primitives, used at each prompt-assembly site:

  replace_or_base(base_template, target) -> the active `<target>.replace.md`
      (last active profile wins) or `base_template` unchanged. Applied to the
      TEMPLATE *before* placeholder substitution, so a replace author keeps the
      base's `{seeds}` / `%%MODE%%` tokens and owns the machine contract.

  additions(target) -> "\n\n" + joined `<target>.add.md` fragments across the
      active profiles in list order, or "". Appended *after* substitution, so a
      fragment may contain any characters (including `{}`) with no str.format
      hazard.

Both degrade to the no-op result (the base, or "") when CUSTOM_PROFILE is unset,
a folder/file is missing, or anything goes wrong — they never raise into a
prompt-assembly path. Read-at-call: a dropped-in fragment takes effect on the
next assembly, and tests can monkeypatch CUSTOM_PROFILE without reimporting.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # braindb/.. -> repo root


def _active_profiles() -> list[str]:
    """Profile names from CUSTOM_PROFILE (comma-separated), in order. [] when unset."""
    raw = os.getenv("CUSTOM_PROFILE", "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()]


def _profiles_root() -> Path:
    base = os.getenv("CUSTOM_PROFILE_DIR", "custom-profiles").strip() or "custom-profiles"
    p = Path(base)
    return p if p.is_absolute() else _REPO_ROOT / p


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except Exception:
        return None


def replace_or_base(base_template: str, target: str) -> str:
    """Return the active `<target>.replace.md` (last profile wins), else
    `base_template` unchanged. Never raises."""
    try:
        root = _profiles_root()
        chosen = base_template
        for name in _active_profiles():
            text = _read(root / name / f"{target}.replace.md")
            if text is not None:
                chosen = text
        return chosen
    except Exception:
        return base_template


def additions(target: str) -> str:
    """Return '\\n\\n' + joined `<target>.add.md` fragments across active profiles
    (list order), or '' when none / on any error. Never raises."""
    try:
        root = _profiles_root()
        frags: list[str] = []
        for name in _active_profiles():
            text = _read(root / name / f"{target}.add.md")
            if text is not None:
                stripped = text.strip()
                if stripped:
                    frags.append(stripped)
        return ("\n\n" + "\n\n".join(frags)) if frags else ""
    except Exception:
        return ""
