"""Section-level operations on wiki markdown bodies.

Wiki bodies live as one markdown blob in `entities.content`. This module
parses, splices, and validates them at the section level so the writer
agent can edit ONE section at a time instead of rewriting the whole
body — the fix for big-wiki context exhaustion on smaller-context
models (see plan: read-write tools / handoff).

Sections are anchored on `<!-- section:NAME -->` HTML-comment markers
that the writer prompt already mandates (see `wiki_writer_prompt.md`
"Recommended structure"). Everything before the first marker is the
HEADER (meta-comment + `# Title` + `> **Summary:** ...` callout) and
is preserved verbatim by all splice operations.

Optimistic concurrency: every read returns the wiki's current
`wikis_ext.revision`. Every write requires the caller to pass that
revision back as `expect_revision`. A mismatch raises
`StaleRevisionError` so the caller re-reads and retries instead of
silently stomping on a concurrent edit.

Pure parsing functions (`parse_sections`, `splice_section`,
`delete_section`, `check_grammar`) are DB-free and unit-testable.
The two DB helpers at the bottom (`fetch_wiki_for_section_op`,
`apply_section_write`) are the only stateful surface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class StaleRevisionError(Exception):
    """Raised by `apply_section_write` when the caller's
    `expect_revision` no longer matches the wiki's current revision.
    Means the body was changed by someone else (or by the same agent
    in an earlier turn) since the caller last read it."""


# Section marker. Captured group = the section name. We accept
# alphanumerics, dashes, and underscores in the name — matches the
# writer prompt's convention (e.g. `overview`, `timeline`,
# `contradictions`, `sources`, `references`).
_MARKER_RE = re.compile(
    r"<!--\s*section:\s*([A-Za-z0-9_\-]+)\s*-->",
    re.MULTILINE,
)

# UUID shape expected right after `[[ref:`. Real wiki bodies use two
# forms — canonical `[[ref:UUID]]` / `[[ref:UUID|display]]` AND a
# grouped variant `[[ref:UUID1], [ref:UUID2]]` that the writer
# occasionally emits and the frontend plan documents as tolerated.
# Rather than enumerate both forms, we just verify that each
# `[[ref:` is followed by a UUID-looking prefix (8 hex + dash). A
# token that fails this minimal check is genuinely broken (truncated,
# corrupted, or fabricated by a confused model).
_UUID_HEAD_RE = re.compile(r"[0-9a-fA-F]{8}-")


@dataclass(frozen=True)
class Section:
    name: str
    content: str  # body text AFTER the marker, up to next marker / EOF

    @property
    def char_count(self) -> int:
        return len(self.content)


def parse_sections(body: str) -> tuple[str, list[Section]]:
    """Split a wiki body into (header, sections).

    `header` = everything before the first marker (verbatim).
    `sections` = ordered list, each carrying its name + content.

    If the body has no markers, returns `(body, [])` — callers handle
    the strict-markers contract themselves.
    """
    matches = list(_MARKER_RE.finditer(body))
    if not matches:
        return body, []
    header = body[: matches[0].start()]
    sections: list[Section] = []
    for i, m in enumerate(matches):
        content_start = m.end()
        # consume the single newline that conventionally follows the
        # marker line, so section content starts on its own line
        if content_start < len(body) and body[content_start] == "\n":
            content_start += 1
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append(Section(
            name=m.group(1),
            content=body[content_start:content_end],
        ))
    return header, sections


def splice_section(body: str, section_name: str, new_content: str) -> str:
    """Replace one named section's content. If the section doesn't exist,
    append a new section at the end of the body with that name.

    `new_content` is the section's text WITHOUT the marker line — this
    function emits the marker. The result is always normalised so the
    rebuilt body parses identically to one written from scratch.
    """
    header, sections = parse_sections(body)
    new_content = new_content.rstrip("\n") + "\n"
    if any(s.name == section_name for s in sections):
        sections = [
            Section(s.name, new_content if s.name == section_name else s.content)
            for s in sections
        ]
        return _rebuild(header, sections)
    # not found → append a fresh section after the last one
    sections = sections + [Section(section_name, new_content)]
    return _rebuild(header, sections)


def append_to_section(body: str, section_name: str, added_content: str) -> str:
    """Append text to the END of one named section, keeping everything
    already in it byte-for-byte.

    This is the incremental case the pipeline actually runs on: adding one
    `[[ref:UUID]]` bullet. `splice_section` requires the caller to supply the
    section's WHOLE new content, which on a large section forces a
    read-everything / re-emit-everything cycle — and since the read side is
    capped (`MAX_OUTPUT_CHARS`), on a section past that cap it cannot be done
    correctly at all. Resolving the append here, from the body the caller
    already holds, means the model never has to carry the prior content.

    A section that doesn't exist yet is created with `added_content`, matching
    `splice_section`'s append-a-new-section behaviour.
    """
    _, sections = parse_sections(body)
    existing = next((s for s in sections if s.name == section_name), None)
    if existing is None:
        return splice_section(body, section_name, added_content)
    merged = existing.content.rstrip("\n") + "\n" + added_content.lstrip("\n")
    return splice_section(body, section_name, merged)


def delete_section(body: str, section_name: str) -> str:
    """Remove the named section (and its marker) from the body.
    Raises KeyError if the section isn't present."""
    header, sections = parse_sections(body)
    remaining = [s for s in sections if s.name != section_name]
    if len(remaining) == len(sections):
        raise KeyError(f"section not found: {section_name}")
    return _rebuild(header, remaining)


def _rebuild(header: str, sections: list[Section]) -> str:
    parts: list[str] = []
    if header:
        parts.append(header if header.endswith("\n") else header + "\n")
    for s in sections:
        parts.append(f"<!-- section:{s.name} -->\n")
        content = s.content if s.content.endswith("\n") else s.content + "\n"
        parts.append(content)
    return "".join(parts)


def check_grammar(body: str) -> list[str]:
    """Return a list of grammar issues with the wiki body. Empty = OK.

    Checked:
    - At least one `<!-- section:X -->` marker exists (strict-markers).
    - No malformed `[[ref:` tokens (i.e. `[[ref:` that doesn't match
      the canonical `[[ref:UUID]]` or `[[ref:UUID|text]]` shape).
    - The `> **Summary:**` callout exists in the header.
    """
    issues: list[str] = []
    header, sections = parse_sections(body)
    if not sections:
        issues.append("no <!-- section:X --> markers (strict-markers contract)")
    for m in re.finditer(r"\[\[ref:", body):
        # Skip past "[[ref:" (6 chars) and check the next chars look like
        # the start of a UUID. Tolerates the grouped form
        # `[[ref:UUID1], [ref:UUID2]]` since we only check the head.
        if not _UUID_HEAD_RE.match(body[m.end():m.end() + 9]):
            issues.append(f"malformed [[ref: token at char offset {m.start()}")
    if "> **Summary:**" not in header:
        issues.append("missing > **Summary:** callout in header")
    return issues


# ====================================================================== #
# DB helpers                                                              #
# ====================================================================== #

def fetch_wiki_for_section_op(conn, wiki_id: str) -> tuple[str, int] | None:
    """Return (content, revision) for the wiki, or None if not found.
    Used by every read-side section tool to capture both the body and
    the current revision token in one query."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT e.content, w.revision
               FROM entities e JOIN wikis_ext w ON w.entity_id = e.id
               WHERE e.id = %s::uuid AND e.entity_type = 'wiki'""",
            (wiki_id,),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else None


def apply_section_write(conn, wiki_id: str, new_body: str,
                         expect_revision: int) -> int:
    """Atomically replace the wiki's content + bump its revision.

    The revision UPDATE is conditional on `revision = expect_revision`,
    so two writers cannot stomp each other. Returns the new revision
    on success. Raises `StaleRevisionError` if the revision didn't
    match — caller should re-read and retry.
    """
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE wikis_ext
                  SET revision = revision + 1,
                      last_synthesised_at = now()
                WHERE entity_id = %s::uuid AND revision = %s
            RETURNING revision""",
            (wiki_id, expect_revision),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT revision FROM wikis_ext WHERE entity_id = %s::uuid",
                (wiki_id,),
            )
            cur_row = cur.fetchone()
            if cur_row is None:
                raise StaleRevisionError(f"wiki not found: {wiki_id}")
            raise StaleRevisionError(
                f"expected revision {expect_revision}, current is {cur_row[0]} "
                f"— re-read the section before retrying"
            )
        new_revision = row[0]
        cur.execute(
            "UPDATE entities SET content = %s WHERE id = %s::uuid",
            (new_body, wiki_id),
        )
        return new_revision
