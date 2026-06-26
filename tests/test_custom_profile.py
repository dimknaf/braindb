"""
Pure-unit coverage for the optional custom-profile prompt composition
(braindb.custom_profile). No live stack, no Docker: these monkeypatch
CUSTOM_PROFILE / CUSTOM_PROFILE_DIR and write tiny fragment files into tmp_path,
so they run with `pytest` even when the stack is down (no `api` fixture).

The load-bearing guarantee is the byte-identical test: with no active profile
the composition is a strict no-op, so the wiki prompts are exactly today's.
"""
import pytest

from braindb import custom_profile


def _write(root, profile, filename, text):
    d = root / profile
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(text, encoding="utf-8")


@pytest.fixture
def profiles_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CUSTOM_PROFILE_DIR", str(tmp_path))
    return tmp_path


def test_unset_is_strict_noop(profiles_dir, monkeypatch):
    monkeypatch.delenv("CUSTOM_PROFILE", raising=False)
    base = "BASE PROMPT {seeds} {wiki_catalog}"
    assert custom_profile.replace_or_base(base, "wiki_maintainer") == base
    assert custom_profile.additions("wiki_maintainer") == ""
    assert custom_profile.additions("wiki_writer") == ""


def test_whitespace_only_profile_is_noop(profiles_dir, monkeypatch):
    monkeypatch.setenv("CUSTOM_PROFILE", "   ")
    base = "BASE"
    assert custom_profile.replace_or_base(base, "wiki_writer") == base
    assert custom_profile.additions("wiki_writer") == ""


def test_maintainer_prompt_byte_identical_when_unset(profiles_dir, monkeypatch):
    """Zero-default-impact proof: the wired assembly equals the bare base."""
    monkeypatch.delenv("CUSTOM_PROFILE", raising=False)
    base = "RULES...\nSEEDS:\n{seeds}\nWIKIS:\n{wiki_catalog}\n"
    bare = base.format(seeds="S", wiki_catalog="C")
    wired = (
        custom_profile.replace_or_base(base, "wiki_maintainer")
        .format(seeds="S", wiki_catalog="C")
        + custom_profile.additions("wiki_maintainer")
    )
    assert wired == bare


def test_add_one_profile(profiles_dir, monkeypatch):
    _write(profiles_dir, "p1", "wiki_writer.add.md", "EXTRA RULE")
    monkeypatch.setenv("CUSTOM_PROFILE", "p1")
    assert custom_profile.additions("wiki_writer") == "\n\nEXTRA RULE"
    # an untouched target stays empty
    assert custom_profile.additions("wiki_maintainer") == ""


def test_add_multiple_profiles_joined_in_order(profiles_dir, monkeypatch):
    _write(profiles_dir, "a", "wiki_writer.add.md", "FROM A")
    _write(profiles_dir, "b", "wiki_writer.add.md", "FROM B")
    monkeypatch.setenv("CUSTOM_PROFILE", "a,b")
    assert custom_profile.additions("wiki_writer") == "\n\nFROM A\n\nFROM B"
    monkeypatch.setenv("CUSTOM_PROFILE", "b,a")
    assert custom_profile.additions("wiki_writer") == "\n\nFROM B\n\nFROM A"


def test_replace_swaps_template_last_wins(profiles_dir, monkeypatch):
    base = "BASE"
    _write(profiles_dir, "a", "wiki_writer.replace.md", "REPLACED BY A")
    _write(profiles_dir, "b", "wiki_writer.replace.md", "REPLACED BY B")
    monkeypatch.setenv("CUSTOM_PROFILE", "a,b")
    assert custom_profile.replace_or_base(base, "wiki_writer") == "REPLACED BY B"
    # a profile without a replace leaves the earlier one in place
    monkeypatch.setenv("CUSTOM_PROFILE", "a,no_replace_here")
    assert custom_profile.replace_or_base(base, "wiki_writer") == "REPLACED BY A"


def test_add_fragment_with_braces_is_format_safe(profiles_dir, monkeypatch):
    """An `.add.md` with literal `{x}` must not break the maintainer's
    `.format(...)`, because additions are appended AFTER substitution."""
    base = "RULES {seeds} {wiki_catalog}"
    _write(profiles_dir, "p", "wiki_maintainer.add.md", "company note: use {ticker} as id")
    monkeypatch.setenv("CUSTOM_PROFILE", "p")
    tmpl = custom_profile.replace_or_base(base, "wiki_maintainer")
    assembled = tmpl.format(seeds="S", wiki_catalog="C") + custom_profile.additions("wiki_maintainer")
    assert assembled == "RULES S C\n\ncompany note: use {ticker} as id"


def test_blank_add_fragment_yields_no_suffix(profiles_dir, monkeypatch):
    _write(profiles_dir, "p", "wiki_writer.add.md", "   \n  ")
    monkeypatch.setenv("CUSTOM_PROFILE", "p")
    assert custom_profile.additions("wiki_writer") == ""


def test_missing_profile_and_dir_never_raise(profiles_dir, monkeypatch):
    monkeypatch.setenv("CUSTOM_PROFILE", "ghost")  # folder doesn't exist under tmp
    assert custom_profile.additions("wiki_writer") == ""
    assert custom_profile.replace_or_base("BASE", "wiki_writer") == "BASE"
    monkeypatch.setenv("CUSTOM_PROFILE_DIR", str(profiles_dir / "does_not_exist"))
    assert custom_profile.additions("wiki_writer") == ""
    assert custom_profile.replace_or_base("BASE", "wiki_writer") == "BASE"
