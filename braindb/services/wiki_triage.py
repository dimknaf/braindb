"""
Wiki triage by JEV: the maintainer's decision for each seed, taken as
one-token picks instead of an agent run.

For each orphan seed the maintainer chooses one action (skip, ambiguous,
consolidate, attach, create) and, for attach, which catalog wiki. That is a
choice from a known set, so it is asked the JEV way (services/jev.py
`Choice`): BrainDB's own recall gathers the seed's evidence in code;
candidate wikis are shortlisted from that evidence (wikis among it, wikis
that summarise it, wikis whose name carries one of the seed's keywords); a
cached guide with worked examples (agent/prompts/wiki_triage.md and
wiki_triage_examples.json) and the seed's evidence are read once; the model
answers once, thinking off, the way the examples do ("action: attach" then
"wiki: A"), and each decision is ONE token of that answer, read as
probabilities over its options where its lead ends. The only words taken
from the answer are a new page's name.

Measured on a news memory against blind two-judge gold (198 past maintainer
decisions, replayed with the catalog and evidence as they stood at each
decision): 84.3% correct, against 80.3% for the maintainer agent. Live,
through this route, it decided every seed at about a sixth of the agent's
cost per seed and in under half its time, most of which is the one recall.

A seed it cannot decide confidently — low probability on the options, an
attach with no fitting candidate, a consolidate with fewer than two
duplicates, an empty or cut-off name, or any failure — is left to the
maintainer agent, exactly as before. On by default (JEV_WIKI_TRIAGE), once
the server passes the probe.

Custom profiles shape the agent only. Their maintainer additions are written
for an agent that researches before it decides; added to this guide, the
news memory's own profile made the one-token decisions skip pages the agent
creates (8 points fewer correct). A profile that replaces the maintainer's
prompt redefines triage itself, so JEV triage stays off under it.
"""
from __future__ import annotations

import json
import logging
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg2.extras

from braindb import custom_profile
from braindb.agent.run_state import add_usage
from braindb.agent.schemas import MaintainerClusterItem
from braindb.config import settings
from braindb.db import get_conn
from braindb.schemas.search import ContextRequest
from braindb.services import jev
from braindb.services.context import assemble_context

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).resolve().parent.parent / "agent" / "prompts"
_GUIDE = (_PROMPTS / "wiki_triage.md").read_text(encoding="utf-8").strip()
_EXAMPLES = json.loads((_PROMPTS / "wiki_triage_examples.json").read_text(encoding="utf-8"))
ACTIONS = ["skip", "ambiguous", "consolidate", "attach", "create"]
LABELS = list(string.ascii_uppercase[:8])  # at most 8 candidate wikis per seed
# Below this much probability on the options the model wanted to answer
# something else: the agent decides that seed instead.
MIN_MASS = jev.MIN_MASS
SAME = 0.5            # P(yes) from which a candidate counts as a duplicate page
EVIDENCE_ITEMS = 12   # evidence lines shown per seed
WORKERS = 4           # seeds of one triage unit decided in parallel

_state: dict = {"choice": None, "options": None, "off_until": 0.0}
_lock = threading.Lock()


def wanted() -> bool:
    """Whether JEV triage serves maintainer runs now."""
    return (settings.jev_wiki_triage
            and time.monotonic() >= _state["off_until"]
            # a profile's `.replace.md` rewrites the maintainer's whole contract
            and not custom_profile.replace_or_base("", "wiki_maintainer"))


def _clip(text: str | None, n: int) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n] + "..."


def _switch_off(error: Exception) -> None:
    _state["off_until"] = time.monotonic() + jev.REPROBE_S
    _state["choice"] = None
    reason = str(error) if isinstance(error, jev.Unavailable) else type(error).__name__
    logger.warning("JEV wiki triage off for %d s: %s", jev.REPROBE_S, reason)


def _choice() -> tuple[jev.Choice, dict]:
    """The cached prompt family and its options, built and probed once per process."""
    with _lock:
        if _state["choice"] is None:
            choice = jev.Choice(_GUIDE, [(e["user"], e["answer"]) for e in _EXAMPLES["examples"]])
            options = {"action": choice.options(ACTIONS),
                       "label": choice.options(LABELS + ["none"]),
                       "yes_no": choice.options(["yes", "no"])}
            # The model must triage a clear-cut seed right, action and wiki;
            # this request also puts the guide in the server's cache.
            probe = _EXAMPLES["probe"]
            reply = choice.reply(choice.encode(probe["user"]))
            probs, mass = choice.ask(reply, "action:", options["action"])
            ok = max(probs, key=probs.get) == probe["action"] and mass >= MIN_MASS
            if ok:
                probs, mass = choice.ask(reply, f"action: {probe['action']}\nwiki:",
                                         {k: options["label"][k] for k in ("A", "B", "none")})
                ok = max(probs, key=probs.get) == probe["wiki"] and mass >= MIN_MASS
            if not ok:
                raise jev.Unavailable("the model does not triage the probe seed reliably")
            _state.update(choice=choice, options=options)
            logger.info("JEV wiki triage on")
        return _state["choice"], _state["options"]


def _brief(conn, seed: dict, active: dict[str, tuple[int, dict]]) -> tuple[str, dict[str, str]]:
    """What the decisions read for one seed, and its candidate label -> wiki id."""
    content = " ".join((seed.get("content") or "").split())
    queries = [content[:200]] + [k for k in (seed.get("keywords") or [])[:3] if k]
    result = assemble_context(conn, ContextRequest(queries=queries, max_results=20,
                                                   include_always_on_rules=False))
    evidence = [it for it in result.items if str(it.id) != str(seed["id"])]
    # Shortlist by code, never by the model: wikis among the evidence, then
    # wikis that summarise evidence items, then wikis named by a keyword.
    shortlist = [str(it.id) for it in evidence if it.entity_type == "wiki" and str(it.id) in active]
    ids = [str(it.id) for it in evidence]
    if ids:
        with conn.cursor() as cur:
            cur.execute("""SELECT r.from_entity_id::text FROM relations r
                           WHERE r.relation_type = 'summarises' AND r.to_entity_id = ANY(%s::uuid[])
                           GROUP BY 1 ORDER BY count(*) DESC""", (ids,))
            shortlist += [w for (w,) in cur.fetchall() if w in active]
    for kw in (seed.get("keywords") or [])[:3]:
        if kw and len(kw) > 3:
            shortlist += [w for w, (_, info) in active.items()
                          if kw.lower() in (info["canonical_name"] or "").lower()]
    labels = dict(zip(LABELS, list(dict.fromkeys(shortlist))))
    summaries: dict[str, str] = {}
    if labels:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id::text AS id, summary FROM entities WHERE id = ANY(%s::uuid[])",
                        (list(labels.values()),))
            summaries = {r["id"]: r["summary"] for r in cur.fetchall()}
    lines = [f"SEED: {_clip(content, 600)}", "EVIDENCE:"]
    lines += [f"- [{it.entity_type}] {_clip(it.title, 80) + ' - ' if it.title else ''}{_clip(it.content, 220)}"
              for it in evidence[:EVIDENCE_ITEMS]]
    lines.append("CANDIDATE WIKIS:")
    lines += [f"[{label}] {_clip(active[w][1]['canonical_name'], 60)} - {_clip(summaries.get(w), 90)} "
              f"({active[w][1]['char_count']}ch)" for label, w in labels.items()] or ["(none)"]
    return "\n".join(lines), labels


def _decide_one(choice: jev.Choice, options: dict, seed_id: str, text: str, labels: dict[str, str],
                active: dict[str, tuple[int, dict]]) -> tuple[MaintainerClusterItem | None, dict]:
    """One seed's decision as the maintainer's, or None (the agent decides)."""
    reply = choice.reply(choice.encode(text))  # every decision on this seed reads this one answer
    usage = {"requests": 1, **jev.usage_of(reply)}

    def ask(lead: str, opts: dict) -> tuple[str, dict, float]:
        probs, mass = choice.ask(reply, lead, opts)
        return max(probs, key=probs.get), probs, mass

    action, probs, mass = ask("action:", options["action"])
    if mass < MIN_MASS:
        return None, usage
    decision = {"entity_id": seed_id, "action": action,
                "rationale": f"JEV triage: {action} (P={probs[action]:.2f})"}
    if action == "attach":
        if not labels:
            return None, usage
        pick, p, m = ask("action: attach\nwiki:",
                         {k: options["label"][k] for k in list(labels) + ["none"]})
        if m < MIN_MASS or pick == "none":
            return None, usage
        number, info = active[labels[pick]]
        decision["target_wiki_no"] = number
        decision["rationale"] += f"; wiki {info['canonical_name']} (P={p[pick]:.2f})"
    elif action == "consolidate":
        same = []
        for label, wiki in labels.items():
            _, p, m = ask(f"action: consolidate\n[{label}] same subject (yes or no):", options["yes_no"])
            if m >= MIN_MASS and p["yes"] >= SAME:
                same.append(active[wiki][0])
        if len(same) < 2:
            return None, usage
        decision["consolidate_nos"] = same
    elif action == "create":
        decision["proposed_name"] = choice.name(reply, "action: create\nname:")
    item = MaintainerClusterItem(**decision)
    if action == "create" and not item.proposed_name:
        return None, usage  # no name, or one the schema reads as none ("None", "n/a")
    return item, usage


def decide(seeds: list[dict], cat: list[dict]) -> dict[str, MaintainerClusterItem]:
    """JEV's decisions for the seeds it can decide, by seed entity id; the
    seeds missing from the result go to the maintainer agent. A failure fails
    open: one seed's to the agent, a capability failure's every seed. Usage
    goes to one log line, and to the caller's usage ledger when it has one."""
    if not wanted() or not seeds:
        return {}
    started = time.monotonic()
    try:
        choice, options = _choice()
    except Exception as e:  # noqa: BLE001 - JEV off; every seed goes to the agent
        _switch_off(e)
        return {}
    active = {w["id"]: (n, w) for n, w in enumerate(cat, start=1)}

    def one(seed: dict) -> tuple[MaintainerClusterItem | None, dict]:
        try:
            with get_conn() as conn:  # its own short transaction
                text, labels = _brief(conn, seed, active)
            return _decide_one(choice, options, str(seed["id"]), text, labels, active)
        except jev.Unavailable as e:
            _switch_off(e)
        except Exception as e:  # noqa: BLE001 - this seed goes to the agent
            logger.warning("JEV wiki triage failed open on a seed: %s", type(e).__name__)
        return None, {}

    briefs = [s["brief"] for s in seeds]
    # The first seed goes alone: on a cold cache it computes the prefix once,
    # and the rest then reuse it instead of all computing it at the same time.
    results = [one(briefs[0])]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results += pool.map(one, briefs[1:])
    decided, total = {}, {"requests": 0, "input": 0, "cached": 0, "output": 0}
    for seed, (decision, usage) in zip(briefs, results):
        if decision is not None:
            decided[str(seed["id"])] = decision
        for k in total:
            total[k] += usage.get(k, 0)
    add_usage("jev", calls=len(seeds), failopen=len(seeds) - len(decided), **total)
    logger.info("JEV wiki triage decided %d/%d seeds in %.1fs (in=%d cached=%d out=%d)",
                len(decided), len(seeds), time.monotonic() - started,
                total["input"], total["cached"], total["output"])
    return decided
