"""
The wiki writer behind a JEV gate, inside the writer's own conversation: of
the facts memory holds about a page's subject, only those JEV keeps reach the
writer, and on a long page only the sections the job's facts belong in. Each
is decided by ONE token in the writer's own first request, so every fact and
section is computed once.

Without the gate the writer researches the subject itself (2-4 recalls of 30
items), runs a mandatory identity subagent over what it gathered, and pages
through a long page's sections to find where new facts go, all of it carried
through every later turn, until a long job hands off to a successor that
reads it again. With JEV_WIKI_GATE on, the router builds the gated writer's
first request: its system prompt (agent/prompts/writer_gated_system.md, with
worked examples of the two questions below, cached for every job), then the
job (subject, members, rules, the page or its lettered outline). `prepare`
appends to it, LAST, the facts BrainDB's own recall turns up for the page (at
most PACK; wikis, keywords, sources, the job's members and the facts the page
already cites left out) and, with JEV_HOPS on, the facts one relation away
from the members (at most PACK_HOPS, services/hops.py), one at a time, each
followed by one question in that same conversation (services/jev.py `ask`:
thinking off, its answer read from the logprobs of ONE token):
- "[f2] | about the subject (yes or no):" yes: the conversation continues
  through the fact; no, or too little probability on the options: it is
  rolled back, and the next fact follows what was kept;
- on a page too long to inline, for every member and kept fact,
  "[m1] | section:" the outline letter of the section it belongs in, or
  "new" (an unsure one is left to the writer).
The members are the writer's to cite whatever the gate says (a member left
uncited is lost), so they are never asked about. The writer's first turn
continues from the server's prefix cache of all of it, and it edits without
research tools (agent.get_gated_writer_agent), so nothing left out can come
back.

Anything the gate cannot do (off, consolidate, a profile that replaces the
writer's prompt, a failed probe, an error or the deadline) gives None, and
the full writer runs exactly as before.
"""
from __future__ import annotations

import logging
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from braindb import custom_profile
from braindb.agent.run_state import add_usage
from braindb.config import settings
from braindb.db import get_conn
from braindb.schemas.search import ContextRequest
from braindb.services import hops, jev, reader
from braindb.services.context import assemble_context
from braindb.services.wiki_jobs import REF_RE, fetch_members, keywords_from_meta, parse_refs

logger = logging.getLogger(__name__)

ID_LEAD = "| about the subject (yes or no):"  # asked after each found fact, behind its anchor
PL_LEAD = "| section:"                         # asked for each member and kept fact of a long page
LETTERS = list(string.ascii_uppercase[:16])   # at most 16 outline sections are offered
KEEP = 0.5         # P(yes) a found fact needs to reach the writer
MIN_MASS = jev.MIN_MASS  # below this the options hold too little probability: no answer
PACK = 8           # facts found by recall that are read
PACK_HOPS = 6      # with JEV_HOPS on, facts one relation away from the members, read after them
FACT_CHARS = 1200  # of each fact's content
DEADLINE_S = 90    # for all of a job's decisions
WORKERS = 8        # placements of one job in parallel (jev's in-flight limit still applies)

_state: dict = {"server": None, "off_until": 0.0}
_lock = threading.Lock()


@dataclass
class Gate:
    """What the writer is given: the job's members, the found facts kept,
    the found facts left out, (long pages) each fact's section, and its
    prompt through the kept facts."""
    members: list[dict]
    found: list[dict]
    excluded: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)  # fact id -> section name, or "new"
    prompt: str = ""

    @property
    def member_ids(self) -> list[str]:
        return [m["id"] for m in self.members]

    def record(self) -> dict:
        """For the wiki_write activity row."""
        return {"members": self.member_ids, "found": [f["id"] for f in self.found],
                "excluded": self.excluded, "sections": self.sections}


def wanted() -> bool:
    """Whether the gate serves writer runs now."""
    return (settings.jev_wiki_gate
            and time.monotonic() >= _state["off_until"]
            # a profile's `.replace.md` rewrites the writer's whole contract
            and not custom_profile.replace_or_base("", "wiki_writer"))


def _clip(text: str | None, n: int) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n] + "..."


def _fact(anchor: str, f: dict) -> str:
    """One fact as the gated writer's prompt shows it, behind its anchor."""
    return (f"[{anchor}] id: {f['id']}\n  type: {f['entity_type']}\n"
            f"  keywords: {f.get('keywords') or []}\n  content: {_clip(f.get('content'), FACT_CHARS)}")


def members_block(members: list[dict]) -> str:
    """The job's members in the gated writer's prompt, each behind its anchor [m1], [m2]..."""
    return "\n".join(_fact(f"m{i}", m) for i, m in enumerate(members, 1)) or "(none)"


def outline(sections: list) -> list[str]:
    """A long page's outline lines: lettered for placement when it has no more
    sections than there are letters."""
    mark = (lambda i: f"{LETTERS[i]}.") if len(sections) <= len(LETTERS) else (lambda i: "-")
    return [f"{mark(i)} {s.name}: {s.char_count} chars - {_clip(REF_RE.sub('', s.content), 110)}"
            for i, s in enumerate(sections)]


def _subject(canonical: str, old_body: str) -> str:
    """The subject the facts are checked against: the page's name and, when it
    has one, its own summary line (a new page's subject is its members)."""
    summary = next((ln.split("**Summary:**", 1)[1] for ln in (old_body or "").splitlines()
                    if "**Summary:**" in ln), "")
    return f"SUBJECT: {canonical}" + (f"\nSUMMARY: {_clip(summary, 400)}" if summary.strip() else "")


def _found(conn, canonical: str, old_body: str, members: list[dict]) -> list[dict]:
    """Facts the writer's own research would turn up: recall on the page's
    name and keywords, minus what the job or the page already holds. Read
    in full: recall's items are previews, whose cut-off marker points at
    tools the gated writer does not have."""
    skip = {m["id"].lower() for m in members} | parse_refs(old_body)
    keywords = keywords_from_meta(old_body) or [k for m in members for k in (m.get("keywords") or [])]
    queries = [canonical] + [k for k in dict.fromkeys(keywords) if k and k != canonical][:3]
    result = assemble_context(conn, ContextRequest(queries=queries, max_results=30, include_always_on_rules=False))
    ids = [str(it.id) for it in result.items
           if it.entity_type in ("fact", "thought") and str(it.id).lower() not in skip][:PACK]
    if settings.jev_hops:  # then the members' own relation neighbours (services/hops.py)
        near = hops.neighbours(conn, [m["id"] for m in members], skip | {i.lower() for i in ids}, PACK_HOPS,
                               types=("fact", "thought"))
        ids += [n["id"] for n in near]
    rows = {r["id"]: r for r in fetch_members(conn, ids)}
    return [rows[i] for i in ids if i in rows]  # recall's order, then the neighbours'


def _switch_off(error: Exception) -> None:
    with _lock:
        _state.update(off_until=time.monotonic() + jev.REPROBE_S, server=None)
    reason = str(error) if isinstance(error, jev.Unavailable) else type(error).__name__
    logger.warning("JEV wiki gate off for %d s: %s", jev.REPROBE_S, reason)


# A made-up job the model must judge right, asked in the writer's own system
# prompt and tools: one fact about the subject, one about another that only
# shares its name.
_PROBE_HEAD = "### Facts found in memory\nSUBJECT: Rowan bakery\nSUMMARY: a bakery on Tarn Street, opened in 2018."
_PROBE_FACTS = ({"id": "probe-1", "entity_type": "fact", "content": "Kel Aster opened the Rowan bakery on Tarn Street in June 2018."},
                {"id": "probe-2", "entity_type": "fact", "content": "Rowan Pell, the Brenno actor, starred in the 1990s sitcom Tarn Lane."})


def _ready(turn: reader.Turn) -> jev.Server:
    """The agent model's server, probed once per process in the gated
    writer's own system prompt and tools (again after REPROBE_S when it was
    switched off)."""
    with _lock:
        if time.monotonic() < _state["off_until"]:  # a run that waited here while the probe failed
            raise jev.Unavailable("the wiki gate is off until it is probed again")
        if _state["server"] is None:
            server = jev.Server()
            try:
                probe = reader.Turn([turn.messages[0], {"role": "user", "content": _PROBE_HEAD}], turn.tools)
                p = []
                for i, f in enumerate(_PROBE_FACTS, 1):
                    content = f"{_PROBE_HEAD}\n\n{_fact(f'f{i}', f)}"
                    probs, mass, _ = jev.ask(server, reader.with_content(probe, content), f"[f{i}] {ID_LEAD}",
                                             jev.YES_NO, probe.tools)
                    if mass < MIN_MASS:
                        raise jev.Unavailable("the options hold too little probability in the probe job")
                    p.append(probs["yes"])
                if p[0] - p[1] < 0.5:
                    raise jev.Unavailable("the model does not judge the probe job's facts reliably")
            except jev.Unavailable:
                raise
            except Exception as e:  # noqa: BLE001 - a probe that cannot finish keeps it off too
                raise jev.Unavailable(type(e).__name__) from None
            _state["server"] = server
            logger.info("JEV wiki gate on")
        return _state["server"]


def prepare(turn: reader.Turn, canonical: str, old_body: str, members: list[dict],
            sections: list) -> Gate | None:
    """The gate for one attach/create job, or None: the full writer runs.
    `turn` is the gated writer's first request with the job as its user
    message (the found facts not yet in it); `sections` are a long page's,
    offered for placement (none for a page shown in full)."""
    if not wanted() or not members:
        return None
    started = time.monotonic()
    usage = {"requests": 0, "input": 0, "cached": 0, "output": 0}
    gate = None
    try:
        server = _ready(turn)
        with get_conn() as conn:  # its own short transaction
            found = _found(conn, canonical, old_body, members)
        # The deadline is for the decisions: recall takes what it takes, as
        # it would for the full writer's own research.
        gate = _decide(server, turn, canonical, old_body, members, found, sections,
                       time.monotonic() + DEADLINE_S, usage)
    except jev.Unavailable as e:
        _switch_off(e)
    except Exception as e:  # noqa: BLE001 - unreachable, refused or too slow: the full writer runs
        logger.warning("JEV wiki gate failed open: %s", type(e).__name__)
    add_usage("jev", calls=1, failopen=int(gate is None), **usage)
    logger.info("JEV wiki gate %s in %.1fs: %s (in=%d cached=%d out=%d)",
                "settled" if gate else "left the job to the full writer", time.monotonic() - started,
                f"{len(gate.members)} members, {len(gate.found)} of {len(gate.found) + len(gate.excluded)} "
                f"found facts kept, {len(gate.sections)} placed" if gate else "-",
                usage["input"], usage["cached"], usage["output"])
    return gate


def _decide(server: jev.Server, turn: reader.Turn, canonical: str, old_body: str, members: list[dict],
            found: list[dict], sections: list, deadline: float, usage: dict) -> Gate:
    placing = bool(sections) and len(sections) <= len(LETTERS)

    def ask(content: str, lead: str, options: list[str]) -> tuple[dict, float, int]:
        """`lead` asked after `content` as the writer's prompt; the prompt's size too."""
        if time.monotonic() > deadline:
            raise TimeoutError("the wiki gate's decisions missed the deadline")
        probs, mass, u = jev.ask(server, reader.with_content(turn, content), lead, options, turn.tools)
        usage["requests"] += 1
        for k in ("input", "cached", "output"):
            usage[k] += u[k]
        return probs, mass, u["input"]

    # The writer's prompt so far: the job, then (last) the facts kept.
    memory = f"{turn.messages[-1]['content']}\n\n### Facts found in memory\n{_subject(canonical, old_body)}"
    gate = Gate(members, [])
    if not found and not placing:
        gate.prompt = memory + "\n(none)"
        return gate
    size = 0  # tokens of the writer's prompt through it, as the server counted the last request
    kept: list[tuple[str, dict]] = []  # (anchor, fact)
    for i, f in enumerate(found, 1):
        fact = f"\n\n{_fact(f'f{i}', f)}"
        if size and size + len(fact) / reader.CHARS_PER_TOKEN + reader.ROOM > reader.WINDOW_TOKENS:
            gate.excluded += [g["id"] for g in found[i - 1:]]
            break
        probs, mass, prompt = ask(memory + fact, f"[f{i}] {ID_LEAD}", jev.YES_NO)
        if mass >= MIN_MASS and probs["yes"] >= KEEP:  # stays: the prompt continues through it
            memory, size = memory + fact, prompt
            kept.append((f"f{i}", f))
        else:                                          # rolled back: the next fact follows `memory`
            gate.excluded.append(f["id"])
    gate.found = [f for _, f in kept]
    if not kept:
        memory += "\n(none)"
    if placing:
        items = [(f"m{i}", m) for i, m in enumerate(members, 1)] + kept
        options = LETTERS[:len(sections)] + ["new"]

        def place(item: tuple[str, dict]) -> tuple[dict, dict, float]:
            probs, mass, _ = ask(memory, f"[{item[0]}] {PL_LEAD}", options)
            return item[1], probs, mass

        # The first alone: it leaves the shared prompt cached for the rest.
        results = [place(items[0])]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results += pool.map(place, items[1:])
        for f, probs, mass in results:
            if mass >= MIN_MASS:  # an unsure placement is left to the writer
                pick = max(probs, key=probs.get)
                gate.sections[f["id"]] = "new" if pick == "new" else sections[LETTERS.index(pick)].name
    gate.prompt = memory
    return gate
