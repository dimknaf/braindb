"""Unit tests for `braindb.services.wiki_sections` — the pure parsing and
splicing layer behind the writer's section-edit tools.

These tests cover the DB-free functions only (`parse_sections`,
`splice_section`, `append_to_section`, `delete_section`, `check_grammar`).
The DB helpers (`fetch_wiki_for_section_op`, `apply_section_write`) are
covered by the end-to-end smoke test inside `braindb_api` (see plan Phase 1).

The contract being tested:

- `parse_sections(body)` returns `(header, [Section(name, content)])`.
  Sections are split on `<!-- section:NAME -->` markers; the header
  is everything before the first marker.
- `splice_section` REPLACES an existing section's content, or APPENDS
  a fresh section if the name is new. Bytes outside the targeted
  section are preserved exactly.
- `append_to_section` ADDS to an existing section without the caller
  supplying its prior content — the genuinely-additive "one more
  citation" case. Existing content is preserved at the content level
  (trailing blank lines collapse to one; markers re-emit canonically):
  no claim or citation can be lost.
- `delete_section` removes a section, raises `KeyError` if missing.
- `check_grammar` flags: no markers, malformed `[[ref:` tokens, missing
  Summary callout. Tolerates the grouped-refs variant `[[ref:UUID1],
  [ref:UUID2]]` documented in the wiki frontend plan.
- Round-trip identity: parse → splice (with same content) → string is
  byte-identical to the input when the input is itself in normal form.
"""
from __future__ import annotations

import pytest

from braindb.services.wiki_sections import (
    Section,
    StaleRevisionError,
    append_to_section,
    check_grammar,
    delete_section,
    parse_sections,
    splice_section,
)

UUID_A = "11111111-1111-1111-1111-111111111111"
UUID_B = "22222222-2222-2222-2222-222222222222"

# A minimal but realistic body in normal form (matches the writer
# prompt's "Recommended structure"). Used as the baseline for splice +
# roundtrip tests.
NORMAL_BODY = (
    "<!-- wiki:meta canonical_name=Test language=en revision=1 -->\n"
    "# Test\n"
    "> **Summary:** one line\n"
    "> **Disambiguation:** what this is\n"
    f"<!-- section:overview -->\n"
    f"opening prose [[ref:{UUID_A}]]\n"
    "<!-- section:timeline -->\n"
    f"2026 — event [[ref:{UUID_B}]]\n"
    "<!-- section:references -->\n"
    f"- [[ref:{UUID_A}]] — source A\n"
    f"- [[ref:{UUID_B}]] — source B\n"
)


# ====================================================================== #
# parse_sections                                                          #
# ====================================================================== #

def test_parse_sections_extracts_each_section_in_order():
    header, sections = parse_sections(NORMAL_BODY)
    names = [s.name for s in sections]
    assert names == ["overview", "timeline", "references"]


def test_parse_sections_preserves_header_verbatim():
    header, _ = parse_sections(NORMAL_BODY)
    assert header.startswith("<!-- wiki:meta")
    assert "# Test" in header
    assert "> **Summary:**" in header
    # header ends at (not after) the first marker
    assert "<!-- section:" not in header


def test_parse_sections_section_content_excludes_marker_line():
    _, sections = parse_sections(NORMAL_BODY)
    overview = next(s for s in sections if s.name == "overview")
    assert overview.content.startswith("opening prose ")
    assert "<!-- section:" not in overview.content


def test_parse_sections_no_markers_returns_empty_sections():
    body = "just plain text with no markers\n"
    header, sections = parse_sections(body)
    assert header == body
    assert sections == []


def test_parse_sections_char_count_is_content_length():
    _, sections = parse_sections(NORMAL_BODY)
    assert all(s.char_count == len(s.content) for s in sections)


# ====================================================================== #
# splice_section — replace existing                                       #
# ====================================================================== #

def test_splice_replace_existing_section():
    new = splice_section(NORMAL_BODY, "overview", "rewritten prose")
    _, sections = parse_sections(new)
    overview = next(s for s in sections if s.name == "overview")
    assert "rewritten prose" in overview.content
    # Other sections untouched
    timeline = next(s for s in sections if s.name == "timeline")
    assert "2026 — event" in timeline.content


def test_splice_replace_preserves_header():
    original_header, _ = parse_sections(NORMAL_BODY)
    new = splice_section(NORMAL_BODY, "overview", "rewritten")
    new_header, _ = parse_sections(new)
    assert new_header == original_header


def test_splice_replace_preserves_section_order():
    new = splice_section(NORMAL_BODY, "timeline", "new timeline")
    _, sections = parse_sections(new)
    assert [s.name for s in sections] == ["overview", "timeline", "references"]


# ====================================================================== #
# splice_section — append new section                                     #
# ====================================================================== #

def test_splice_append_new_section_when_name_missing():
    new = splice_section(NORMAL_BODY, "roadmap", "Q3 2026 plans")
    _, sections = parse_sections(new)
    assert "roadmap" in [s.name for s in sections]
    # appended at the END
    assert sections[-1].name == "roadmap"
    assert "Q3 2026 plans" in sections[-1].content


def test_splice_append_does_not_disturb_existing_sections():
    new = splice_section(NORMAL_BODY, "roadmap", "future")
    _, sections = parse_sections(new)
    # original 3 sections still present in same order
    original_names = ["overview", "timeline", "references"]
    assert [s.name for s in sections][:3] == original_names


# ====================================================================== #
# delete_section                                                          #
# ====================================================================== #

def test_delete_section_removes_named_section():
    new = delete_section(NORMAL_BODY, "timeline")
    _, sections = parse_sections(new)
    names = [s.name for s in sections]
    assert "timeline" not in names
    assert names == ["overview", "references"]


def test_delete_section_raises_keyerror_for_missing():
    with pytest.raises(KeyError):
        delete_section(NORMAL_BODY, "nonexistent")


def test_delete_section_preserves_header():
    original_header, _ = parse_sections(NORMAL_BODY)
    new = delete_section(NORMAL_BODY, "timeline")
    new_header, _ = parse_sections(new)
    assert new_header == original_header


# ====================================================================== #
# Round-trip identity                                                     #
# ====================================================================== #

def test_roundtrip_identity_on_normal_body():
    """Splicing a section with its own content must produce a body that
    is byte-identical to the input. This is the strongest proof that
    the parser + rebuilder are self-consistent — no drift, no marker
    corruption."""
    _, sections = parse_sections(NORMAL_BODY)
    overview = next(s for s in sections if s.name == "overview")
    roundtrip = splice_section(
        NORMAL_BODY, "overview", overview.content.rstrip("\n"),
    )
    assert roundtrip == NORMAL_BODY


# ====================================================================== #
# check_grammar                                                           #
# ====================================================================== #

def test_grammar_clean_body_passes():
    assert check_grammar(NORMAL_BODY) == []


def test_grammar_flags_missing_markers():
    body = "# Test\n> **Summary:** s\nNo markers here.\n"
    issues = check_grammar(body)
    assert any("no <!-- section:" in i for i in issues)


def test_grammar_flags_missing_summary():
    body = (
        "<!-- wiki:meta canonical_name=X -->\n"
        "# X\n"
        "<!-- section:overview -->\n"
        "no summary callout above\n"
    )
    issues = check_grammar(body)
    assert any("> **Summary:**" in i for i in issues)


def test_grammar_tolerates_grouped_refs():
    """The grouped form `[[ref:UUID1], [ref:UUID2]]` is documented in the
    wiki frontend plan as a real-world variant the renderer accepts.
    check_grammar must not flag it as malformed."""
    body = (
        "<!-- wiki:meta canonical_name=X -->\n"
        "# X\n"
        "> **Summary:** s\n"
        "<!-- section:overview -->\n"
        f"grouped citation [[ref:{UUID_A}], [ref:{UUID_B}]] in text\n"
    )
    issues = check_grammar(body)
    # No malformed-ref complaints (the only issue could be summary, but
    # we included it)
    assert not any("malformed" in i for i in issues), issues


def test_grammar_flags_truly_broken_ref():
    body = (
        "<!-- wiki:meta canonical_name=X -->\n"
        "# X\n"
        "> **Summary:** s\n"
        "<!-- section:overview -->\n"
        "broken ref [[ref:not-a-uuid]] here\n"
    )
    issues = check_grammar(body)
    assert any("malformed" in i for i in issues), issues


# ====================================================================== #
# StaleRevisionError class                                                #
# ====================================================================== #

def test_stale_revision_error_is_exception():
    """The DB helpers raise this when expect_revision mismatches the
    current DB revision. The tool wrappers translate it into a string
    error the LLM can read; the class itself is the integration point."""
    assert issubclass(StaleRevisionError, Exception)
    err = StaleRevisionError("expected 5, current 6")
    assert "5" in str(err) and "6" in str(err)


# ====================================================================== #
# Section dataclass                                                       #
# ====================================================================== #

def test_section_is_frozen_dataclass():
    s = Section(name="x", content="y")
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        s.name = "z"  # type: ignore[misc]


def test_section_char_count_property():
    s = Section(name="x", content="abcdef")
    assert s.char_count == 6


# ====================================================================== #
# append_to_section — the incremental add                                 #
# ====================================================================== #
#
# The pipeline's real unit of work is "add one [[ref:UUID]] bullet". Doing
# that through `splice_section` requires the caller to re-emit the section's
# WHOLE content, which on a section past the read cap cannot be done
# correctly at all. These tests pin the property that makes appending safe:
# whatever was already in the section survives byte-for-byte, without the
# caller ever having to hold it.

def test_append_preserves_existing_content_byte_for_byte():
    before = next(s for s in parse_sections(NORMAL_BODY)[1]
                  if s.name == "references")
    out = append_to_section(NORMAL_BODY, "references", "- [[ref:%s]] — source C" % UUID_A)
    after = next(s for s in parse_sections(out)[1] if s.name == "references")
    # every original line is still present, in order, unmodified
    original_lines = [ln for ln in before.content.splitlines() if ln.strip()]
    after_lines = [ln for ln in after.content.splitlines() if ln.strip()]
    assert after_lines[:len(original_lines)] == original_lines


def test_append_adds_the_new_text_at_the_end():
    out = append_to_section(NORMAL_BODY, "references", "- new tail line")
    section = next(s for s in parse_sections(out)[1] if s.name == "references")
    assert section.content.rstrip("\n").endswith("- new tail line")


def test_append_does_not_touch_other_sections_or_header():
    out = append_to_section(NORMAL_BODY, "references", "- new tail line")
    hdr_before, secs_before = parse_sections(NORMAL_BODY)
    hdr_after, secs_after = parse_sections(out)
    assert hdr_after == hdr_before
    untouched_before = {s.name: s.content for s in secs_before if s.name != "references"}
    untouched_after = {s.name: s.content for s in secs_after if s.name != "references"}
    assert untouched_after == untouched_before


def test_append_keeps_section_order():
    out = append_to_section(NORMAL_BODY, "overview", "more prose")
    assert [s.name for s in parse_sections(out)[1]] == [
        "overview", "timeline", "references"]


def test_append_to_missing_section_creates_it_like_splice():
    out = append_to_section(NORMAL_BODY, "sources", "narrative provenance")
    names = [s.name for s in parse_sections(out)[1]]
    assert names == ["overview", "timeline", "references", "sources"]
    created = next(s for s in parse_sections(out)[1] if s.name == "sources")
    assert "narrative provenance" in created.content


def test_append_result_is_reparseable_normal_form():
    out = append_to_section(NORMAL_BODY, "references", "- another")
    # a second append must behave identically on the result of the first
    out2 = append_to_section(out, "references", "- and another")
    section = next(s for s in parse_sections(out2)[1] if s.name == "references")
    assert "- another" in section.content
    assert "- and another" in section.content
    assert check_grammar(out2) == []


def test_append_never_loses_refs_on_a_large_section():
    """The regression this exists for: a section far bigger than the tool
    read cap (8000) must survive an append intact, because the caller never
    supplies its prior content."""
    big = "\n".join(f"- [[ref:{UUID_A}]] — line {i}" for i in range(1200))
    body = splice_section(NORMAL_BODY, "references", big)
    assert len(next(s for s in parse_sections(body)[1]
                    if s.name == "references").content) > 8000
    out = append_to_section(body, "references", f"- [[ref:{UUID_B}]] — new")
    after = next(s for s in parse_sections(out)[1] if s.name == "references")
    assert after.content.count("[[ref:") == 1201
    assert "line 0" in after.content and "line 1199" in after.content
