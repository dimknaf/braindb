"""
The wiki writer behind a JEV gate: what reaches the writer's context is
decided by one-token picks before it starts.

Without the gate the writer researches the subject itself (2-4 recalls of 30
items), runs a mandatory identity subagent over everything it gathered, and
pages through the sections of a long page to find where new facts go — all of
it carried through every later turn, until a long job hands off to a
successor that reads it again. Those are choices from known sets, so with
JEV_WIKI_GATE on they are taken the JEV way (services/jev.py `Choice`: a
cached guide + worked examples, thinking off, ONE token read as probabilities
over the options):
- evidence: BrainDB's own recall on the page's name and keywords, minus
  wikis, keywords, sources, the job's members and the facts the page already
  cites (PACK at most, read in full);
- identity: for every member and every found fact, about the SUBJECT? yes /
  no / ambiguous; a found fact goes on only with P(yes) >= KEEP, and the job
  is settled only when EVERY member passes (a member must be cited wherever
  it is written, so a doubtful one is the full writer's to resolve);
- placement (attach, a page too long to inline): for every kept fact, which
  outline section it belongs in, or a new one.
The writer then edits with exactly those facts and sections in front of it
and no research tools (agent.get_gated_writer_agent), so its context stays
short and nothing the gate left out can come back.

Replayed on 24 past writes of the largest personal memory (hard identity
cases oversampled) against blind two-judge gold: it settled 14 — the rest,
mostly with a doubtful member, went to the full writer — and the facts it
kept carried 3.0 wrong inclusions per 100, against 5.0 for what the
production writer cited of the same facts; placement was acceptable to both
judges for 10 of 10; about $0.0006 per job.

Anything the gate cannot settle — a member not kept, a capability failure,
the deadline — gives None, and the full writer runs exactly as before. Off unless
JEV_WIKI_GATE is on with a self-hosted server, and never under a profile that
replaces the writer's prompt; consolidate jobs always run the full writer.
"""
from __future__ import annotations

import json
import logging
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from braindb import custom_profile
from braindb.agent.run_state import add_usage
from braindb.config import settings
from braindb.db import get_conn
from braindb.schemas.search import ContextRequest
from braindb.services import jev
from braindb.services.context import assemble_context
from braindb.services.wiki_jobs import fetch_members, keywords_from_meta, parse_refs
from braindb.services.wiki_sections import parse_sections

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).resolve().parent.parent / "agent" / "prompts"
_IDENTITY = (_PROMPTS / "wiki_identity.md").read_text(encoding="utf-8").strip()
_PLACEMENT = (_PROMPTS / "wiki_placement.md").read_text(encoding="utf-8").strip()
_EXAMPLES = json.loads((_PROMPTS / "wiki_gate_examples.json").read_text(encoding="utf-8"))
# Each request ends with its lead; the worked answers continue the same lead
# (without that the options held ~2% of the probability on every pick).
ID_LEAD = "about the SUBJECT (yes, no or ambiguous):"
PL_LEAD = "section:"
LETTERS = list(string.ascii_uppercase[:16])  # at most 16 outline sections are offered
KEEP = 0.5        # P(yes) a fact needs to reach the writer
MIN_MASS = 0.5    # below this the options hold too little probability: no answer
PACK = 8          # facts found by recall that are checked
DEADLINE_S = 90   # for all of a job's picks
WORKERS = 8       # picks of one job in parallel (jev's in-flight limit still applies)
_REF = re.compile(r"\[\[ref:[^\]]*\]\]")

_state: dict = {"identity": None, "placement": None, "options": None, "off_until": 0.0}
_lock = threading.Lock()


@dataclass
class Gate:
    """What the writer is given: the job's members (all kept) and the found
    facts kept, the found facts the identity check left out, and (long pages)
    each kept fact's section."""
    members: list[dict]
    found: list[dict]
    excluded: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)  # fact id -> section name, or "new"

    @property
    def member_ids(self) -> list[str]:
        return [m["id"] for m in self.members]

    def record(self) -> dict:
        """For the wiki_write activity row."""
        return {"members": self.member_ids, "found": [f["id"] for f in self.found],
                "excluded": self.excluded, "sections": self.sections}


def wanted() -> bool:
    """Whether the gate serves writer runs now."""
    return (settings.jev_wiki_gate and settings.self_hosted
            and time.monotonic() >= _state["off_until"]
            # a profile's `.replace.md` rewrites the writer's whole contract
            and not custom_profile.replace_or_base("", "wiki_writer"))


def _clip(text: str | None, n: int) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n] + "..."


def _fact(f: dict, n: int = 1200) -> str:
    return f"[{f['entity_type']}] {_clip(f.get('content'), n)}"


def _switch_off(error: Exception) -> None:
    _state["off_until"] = time.monotonic() + jev.REPROBE_S
    _state["identity"] = None
    reason = str(error) if isinstance(error, jev.Unavailable) else type(error).__name__
    logger.warning("JEV wiki gate off for %d s: %s", jev.REPROBE_S, reason)


def _choices() -> tuple[jev.Choice, jev.Choice, dict]:
    """Both prompt families and their options, built and probed once per process."""
    with _lock:
        if time.monotonic() < _state["off_until"]:  # a run that waited here while the probe failed
            raise jev.Unavailable("the wiki gate is off until it is probed again")
        if _state["identity"] is None:
            identity = jev.Choice(_IDENTITY, [(e["user"], f"{ID_LEAD} {e['answer']}") for e in _EXAMPLES["identity"]])
            placement = jev.Choice(_PLACEMENT, [(e["user"], f"{PL_LEAD} {e['answer']}") for e in _EXAMPLES["placement"]])
            options = {"identity": identity.options(["yes", "no", "ambiguous"]),
                       "placement": placement.options(LETTERS + ["new"])}
            # The model must get clear-cut picks right; these requests also
            # put both prefixes in the server's cache.
            probe = _EXAMPLES["probe"]
            picks = [_pick(identity, identity.encode(p["user"]), ID_LEAD, options["identity"])[:2]
                     for p in probe["identity"]]
            pl = probe["placement"]
            picks.append(_pick(placement, placement.encode(pl["user"]), PL_LEAD,
                               {k: options["placement"][k] for k in ("A", "B", "C", "D", "new")})[:2])
            wants = [p["answer"] for p in probe["identity"]] + [pl["answer"]]
            if any(got != want or mass < MIN_MASS for (got, mass), want in zip(picks, wants)):
                raise jev.Unavailable("the model does not answer the gate's probe picks reliably")
            _state.update(identity=identity, placement=placement, options=options)
            logger.info("JEV wiki gate on: prefixes of %d (identity) and %d (placement) tokens",
                        len(identity.g), len(placement.g))
        return _state["identity"], _state["placement"], _state["options"]


def _pick(choice: jev.Choice, body: list[int], lead: str, options: dict) -> tuple[str, float, dict, dict]:
    probs, mass, usage = choice.ask(body, lead, options)
    return max(probs, key=probs.get), mass, probs, usage


def _picks(choice: jev.Choice, texts: list[str], lead: str, options: dict, usage: dict,
           deadline: float) -> list[tuple[str, float, dict]]:
    """One pick per text: the first alone (on a cold cache it computes the
    prefix once), the rest in parallel. Usage is added to `usage`."""
    def one(text: str) -> tuple[str, float, dict, dict]:
        if time.monotonic() > deadline:
            raise TimeoutError("the wiki gate's picks missed the deadline")
        return _pick(choice, choice.encode(text), lead, options)

    results = [one(texts[0])] if texts else []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results += pool.map(one, texts[1:])
    for *_, u in results:
        usage["requests"] += 1
        for k in ("input", "cached", "output"):
            usage[k] += u[k]
    return [(pick, mass, probs) for pick, mass, probs, _ in results]


def _subject(canonical: str, old_body: str, members: list[dict]) -> str:
    """The SUBJECT as the identity question states it: the page's own
    summary line when it has one, else the job's member facts."""
    summary = next((ln.split("**Summary:**", 1)[1] for ln in (old_body or "").splitlines()
                    if "**Summary:**" in ln), "")
    if summary.strip():
        return f"SUBJECT: {canonical}\nSUMMARY: {_clip(summary, 400)}"
    seeds = "\n".join(f"- {_fact(m, 400)}" for m in members[:8])
    return f"SUBJECT: {canonical}\nSEED FACTS (assigned to this page):\n{seeds}"


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
    rows = {r["id"]: r for r in fetch_members(conn, ids)}
    return [rows[i] for i in ids if i in rows]  # recall's order


def prepare(mode: str, canonical: str, old_body: str, members: list[dict], long_page: bool) -> Gate | None:
    """The gate for one writer job, or None: the full writer runs."""
    if not wanted() or mode not in ("attach", "create") or not members:
        return None
    started = time.monotonic()
    usage = {"requests": 0, "input": 0, "cached": 0, "output": 0}
    gate = None
    try:
        identity, placement, options = _choices()
    except Exception as e:  # noqa: BLE001 - the probe failed: off until re-probed
        _switch_off(e)
        identity = None
    try:
        if identity is not None:
            deadline = started + DEADLINE_S
            with get_conn() as conn:  # its own short transaction
                found = _found(conn, canonical, old_body, members)
            subject = _subject(canonical, old_body, members)
            # One line break before FACT, as in the worked examples (Choice.encode squeezes blank lines).
            texts = [f"{subject}\nFACT: {_fact(f)}" for f in members + found]
            picks = _picks(identity, texts, ID_LEAD, options["identity"], usage, deadline)
            kept = [mass >= MIN_MASS and probs["yes"] >= KEEP for _, mass, probs in picks]
            # A doubtful member is the full writer's to resolve: wherever a
            # member is written it must be cited, so leaving one out here
            # would only send it round again.
            if all(kept[:len(members)]):
                found_kept = [f for f, k in zip(found, kept[len(members):]) if k]
                excluded = [f["id"] for f, k in zip(found, kept[len(members):]) if not k]
                gate = Gate(members, found_kept, excluded)
                _, sections = parse_sections(old_body) if long_page else ("", [])
                # Placement offers every section or none: a fact whose home is
                # past the last letter must not be told "new".
                if sections and len(sections) <= len(LETTERS):
                    outline = "\n".join(f"{LETTERS[i]}. {s.name} - {_clip(_REF.sub('', s.content), 110)}"
                                        for i, s in enumerate(sections))
                    placed = members + found_kept
                    opts = {k: options["placement"][k] for k in LETTERS[:len(sections)] + ["new"]}
                    places = _picks(placement, [f"OUTLINE:\n{outline}\nFACT: {_fact(f)}" for f in placed],
                                    PL_LEAD, opts, usage, deadline)
                    for f, (pick, mass, _) in zip(placed, places):
                        if mass >= MIN_MASS:  # an unsure placement is left to the writer
                            gate.sections[f["id"]] = "new" if pick == "new" else sections[LETTERS.index(pick)].name
    except jev.Unavailable as e:
        _switch_off(e)
        gate = None
    except Exception as e:  # noqa: BLE001 - the full writer runs
        logger.warning("JEV wiki gate failed open: %s", type(e).__name__)
        gate = None
    add_usage("jev", calls=1, failopen=int(gate is None), **usage)
    logger.info("JEV wiki gate %s in %.1fs: %s (in=%d cached=%d out=%d)",
                "settled" if gate else "left the job to the full writer", time.monotonic() - started,
                f"{len(gate.members)} members + {len(gate.found)} found kept, {len(gate.excluded)} left out, "
                f"{len(gate.sections)} placed" if gate else "-",
                usage["input"], usage["cached"], usage["output"])
    return gate
