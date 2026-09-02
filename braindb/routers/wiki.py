"""
Wiki pipeline endpoints: cron / maintain / write / jobs.

Stage 1 is manual (no scheduler) — these endpoints are driven by hand or by
the Stage-2 `wiki_scheduler` sidecar. `/cron` and `/jobs` are pure SQL and
non-destructive; `/maintain` and `/write` (later steps) drive the existing
agent endpoint.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Query

from braindb import custom_profile
from braindb.agent.agent import run_typed, get_maintainer_agent, get_writer_agent
from braindb.agent.run_state import install_handoff_slot, release_handoff_slot
from braindb.agent.schemas import MaintainerClusterDecision, WikiWriteResult
from braindb.config import settings
from braindb.db import get_conn
from braindb.services.activity_log import log_activity
from braindb.services import wiki_jobs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/wiki", tags=["wiki"])

_PROMPTS = Path(__file__).parent.parent / "agent" / "prompts"
_MAINTAINER_PROMPT = (_PROMPTS / "wiki_maintainer_prompt.md").read_text(encoding="utf-8")
_WRITER_PROMPT = (_PROMPTS / "wiki_writer_prompt.md").read_text(encoding="utf-8")


def _now_line() -> str:
    """One neutral line giving the model the current date, so temporal reasoning
    (recency, 'as of', dated/bucketed sections) has an anchor. Appended to every
    maintainer/writer prompt; harmless when a subject has no time element."""
    return (
        "\n\n---\n**Current date (UTC):** "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} — treat this as \"now\" when "
        "judging recency, dating claims, or bucketing events.\n"
    )


# A wiki body that carries no real page content: the empty string, whitespace,
# or a small-model serialization artifact echoed verbatim (`""`, `body=""`,
# `{}`). The regex matches ONLY an optional `body=`/`body:` prefix followed by
# quotes/braces/whitespace to the end — a genuine body (meta header + markdown)
# always has real characters, so it never matches. The single definition the
# router uses to decide a body is unpersistable.
_BLANK_BODY = re.compile(r'^\s*(?:body\s*[:=]\s*)?["\'{}\s]*$', re.IGNORECASE)


def _is_blank_body(body: str | None) -> bool:
    return bool(_BLANK_BODY.match(body or ""))


@router.post("/cron")
def wiki_cron():
    """Read-only orphan scan; enqueues one `triage` job per orphan. Idempotent."""
    with get_conn() as conn:
        result = wiki_jobs.run_cron(conn)
        log_activity(conn, "wiki_cron", None, None, details=result)
        return result


@router.post("/maintain")
async def wiki_maintain():
    """
    Process ONE triage unit. Claims the highest-importance pending triage job
    and — when WIKI_TRIAGE_CLUSTER is on and that seed is a non-hub keyword —
    the other pending triage jobs whose entity shares a source fact with it
    (`claim_related_triage`). The maintainer decides EVERY seed in ONE agent
    call; each seed's own triage job is then closed with its own status
    (skip→rejected, attach→done+suggestion, …), exactly as in the
    one-at-a-time path. A singleton (or flag off) is just a 1-seed unit.
    """
    # 1. Claim the unit (primary + related) + per-seed staleness guard — one txn.
    with get_conn() as conn:
        primary = wiki_jobs.claim_one_triage(conn)
        if not primary:
            return {"claimed": 0, "message": "no pending triage jobs"}
        batch_id = str(primary["batch_id"]) if primary["batch_id"] else None
        primary_id = primary["entity_ids"][0]
        primary_brief = wiki_jobs.fetch_entity_brief(conn, primary_id)

        jobs = [primary]
        if (wiki_jobs.WIKI_TRIAGE_CLUSTER and primary_brief
                and primary_brief["entity_type"] == "keyword"):
            jobs += wiki_jobs.claim_related_triage(
                conn, primary_id, str(primary["id"]),
                wiki_jobs.WIKI_TRIAGE_CLUSTER_MAX - 1)

        # Per seed: brief + stale-skip. A stale/missing seed closes its OWN job
        # now (done/failed) with NO LLM call — same as the one-at-a-time path.
        seeds: list[dict] = []
        for j in jobs:
            jid, eid = str(j["id"]), j["entity_ids"][0]
            brief = (primary_brief if eid == primary_id
                     else wiki_jobs.fetch_entity_brief(conn, eid))
            if not brief:
                wiki_jobs.finish_job(conn, jid, "failed", "orphan entity not found")
                continue
            if not wiki_jobs.is_orphan(conn, eid, exclude_triage_job_id=jid):
                wiki_jobs.finish_job(conn, jid, "done",
                                     "already covered — absorbed by a wiki")
                continue
            seeds.append({"job_id": jid, "brief": brief})

        if not seeds:
            return {"claimed": len(jobs), "result": "skipped_stale"}

        # Catalog the model references BY NUMBER; this in-request list IS the
        # numbering used to resolve its chosen number(s) back to ids below.
        cat = wiki_jobs.list_active_wikis(conn)

    # 2. ONE agent call for all live seeds. The prompt directs it to RESEARCH
    #    EACH seed's own subject (recall_memory / view_tree / delegate) before
    #    deciding — co-occurrence is not identity. Returns one decision per seed.
    catalog_txt = (
        "\n".join(
            f"{i}. {w['canonical_name']} ({w['char_count']}ch)"
            for i, w in enumerate(cat, 1)
        )
        or "(no existing wikis yet — attach/consolidate are impossible; "
           "use create/skip/ambiguous)"
    )
    # Optional custom-profile shaping: a `.replace.md` may swap the template
    # (before `.format`), and `.add.md` fragments are appended AFTER `.format`
    # (so a fragment's literal `{`/`}` can never reach str.format). No active
    # profile -> both are no-ops and this prompt is byte-identical to the base.
    _maintainer_tmpl = custom_profile.replace_or_base(_MAINTAINER_PROMPT, "wiki_maintainer")
    prompt = _maintainer_tmpl.format(
        seeds=_seeds_block(seeds), wiki_catalog=catalog_txt
    ) + _now_line() + custom_profile.additions("wiki_maintainer")
    # `run_typed` returns an SDK-validated MaintainerClusterDecision, or raises
    # if the model never submitted — handled below like any agent failure.
    try:
        res: MaintainerClusterDecision = await run_typed(
            prompt, get_maintainer_agent(), MaintainerClusterDecision, max_turns=30
        )
    except Exception as e:
        logger.exception("maintainer agent failed")
        with get_conn() as conn:
            wiki_jobs.finish_jobs(conn, [s["job_id"] for s in seeds], "failed",
                                  f"agent error: {e}"[:500])
            # A crashed run must be visible in the DB audit trail too, not
            # only in wiki_job.last_error and the container log.
            log_activity(conn, "wiki_maintain", None, None, details={
                "result": "failed",
                "jobs": [s["job_id"] for s in seeds],
                "error": str(e)[:500],
            })
        return {"claimed": len(jobs), "result": "failed", "reason": str(e)}

    by_id = {d.entity_id: d for d in res.decisions}

    # 3. Per seed: apply its decision + close its OWN job (same resolution as the
    #    one-at-a-time path). A seed the model omitted is released for a normal
    #    retry (never dropped). One transaction.
    outcomes: dict[str, dict] = {}
    with get_conn() as conn:
        try:
            for s in seeds:
                eid = s["brief"]["id"]
                d = by_id.get(eid)
                if d is None:
                    wiki_jobs.release_or_fail_jobs(
                        conn, [s["job_id"]], "maintainer omitted this seed")
                    outcomes[eid] = {"action": "released_omitted"}
                    continue
                outcomes[eid] = _apply_decision(
                    conn, decision=d.model_dump(), orphan=s["brief"],
                    job_id=s["job_id"], cat=cat, batch_id=batch_id)
        except Exception:
            logger.exception("maintainer persistence failed")
            raise

    return {"claimed": len(jobs), "seeds": len(seeds), "result": outcomes}


def _seeds_block(seeds: list[dict]) -> str:
    """Render the claimed seeds as a list the maintainer decides one-by-one.
    Each carries its entity_id — the decision must echo it back."""
    out = []
    for s in seeds:
        b = s["brief"]
        out.append(
            f"- entity_id: {b['id']}\n"
            f"  entity_type: {b['entity_type']}\n"
            f"  keywords: {b.get('keywords') or []}\n"
            f"  summary: {b.get('summary')}\n"
            f"  content: {(b.get('content') or '')[:4000]}"
        )
    return "\n".join(out)


def _apply_decision(conn, *, decision: dict, orphan: dict, job_id: str,
                    cat: list[dict], batch_id: str | None) -> dict:
    """Resolve ONE maintainer decision for one seed and close THAT seed's triage
    job — the exact attach/create/consolidate/skip handling used in the
    one-at-a-time path, applied per seed. Wikis are referenced by catalog NUMBER
    (`cat` is the numbering)."""
    orphan_id = orphan["id"]
    action = decision.get("action")
    rationale = decision.get("rationale")

    if action in ("skip", "ambiguous"):
        # 'ambiguous' = the data cannot disambiguate identity/scope; the LLM
        # correctly refuses to mint a confident page. Treated as a deliberate
        # skip (self-clears via run_cron).
        wiki_jobs.finish_job(conn, job_id, "rejected", rationale)
        outcome = {"action": action}

    elif action == "attach":
        no = decision.get("target_wiki_no")
        target = (cat[no - 1]["id"]
                  if isinstance(no, int) and 1 <= no <= len(cat)
                  else None)
        if not target or not _is_wiki(conn, target):
            wiki_jobs.finish_job(
                conn, job_id, "failed",
                f"attach: target_wiki_no {no!r} not a valid catalog number (1..{len(cat)})")
            outcome = {"action": "attach", "error": "invalid target_wiki_no"}
        else:
            key = wiki_jobs.suggestion_dedupe_key("attach", target, [orphan_id], [])
            sid = wiki_jobs.insert_suggestion(
                conn, job_type="attach", target_wiki_id=target,
                entity_ids=[orphan_id], dedupe_key=key, rationale=rationale,
                proposed_name=None, batch_id=batch_id)
            wiki_jobs.finish_job(conn, job_id, "done", rationale)
            outcome = {"action": "attach", "suggestion_id": sid, "target_wiki_id": target}

    elif action == "create":
        name = decision.get("proposed_name")
        if not name:
            wiki_jobs.finish_job(conn, job_id, "failed", "create missing proposed_name")
            outcome = {"action": "create", "error": "missing proposed_name"}
        else:
            key = wiki_jobs.suggestion_dedupe_key(
                "create", None, [orphan_id], [], proposed_name=name)
            sid = wiki_jobs.insert_suggestion(
                conn, job_type="create", target_wiki_id=None,
                entity_ids=[orphan_id], dedupe_key=key, rationale=rationale,
                proposed_name=name, batch_id=batch_id)
            wiki_jobs.finish_job(conn, job_id, "done", rationale)
            outcome = {"action": "create", "suggestion_id": sid, "proposed_name": name}

    elif action == "consolidate":
        nos = decision.get("consolidate_nos") or []
        ids = [cat[n - 1]["id"] for n in nos
               if isinstance(n, int) and 1 <= n <= len(cat)]
        wiki_ids = list(dict.fromkeys(ids))  # dedupe, keep order
        if len(wiki_ids) < 2:
            wiki_jobs.finish_job(
                conn, job_id, "failed",
                f"consolidate: need >=2 valid catalog numbers, got {nos!r} (1..{len(cat)})")
            outcome = {"action": "consolidate", "error": "need >=2 valid catalog numbers"}
        else:
            key = wiki_jobs.suggestion_dedupe_key("consolidate", None, [], wiki_ids)
            sid = wiki_jobs.insert_suggestion(
                conn, job_type="consolidate", target_wiki_id=None,
                entity_ids=wiki_ids, dedupe_key=key, rationale=rationale,
                proposed_name=None, batch_id=batch_id)
            # The orphan itself is still unconnected; closing 'done' lets the
            # next cron re-triage it after the merge.
            wiki_jobs.finish_job(conn, job_id, "done", rationale)
            outcome = {"action": "consolidate", "suggestion_id": sid, "wiki_ids": wiki_ids}

    else:
        wiki_jobs.finish_job(conn, job_id, "failed", f"unknown action {action!r}")
        outcome = {"action": action, "error": "unknown action"}

    log_activity(conn, "wiki_maintain", orphan["entity_type"], orphan_id,
                 details={"job_id": job_id, **outcome})
    return outcome


def _members_block(members: list[dict]) -> str:
    if not members:
        return "(none)"
    out = []
    for m in members:
        out.append(
            f"- id: {m['id']}\n  type: {m['entity_type']}\n"
            f"  keywords: {m.get('keywords') or []}\n"
            f"  content: {(m.get('content') or '')[:1200]}"
        )
    return "\n".join(out)


# Above this character count, an attach's `%%CURRENT_BODY%%` is replaced
# by a stub pointing at the section-edit tools. Keeps the writer's
# INITIAL prompt small so it never bumps into the model window before
# it can run a single tool — the section tools (Phase 1) are designed
# exactly for navigating a body without inlining it.
_INLINE_BODY_MAX_CHARS = 4000

# Above this character count the page is already long enough that adding more
# prose costs more than it informs (~10k tokens). The stub gains one paragraph
# telling the writer to stay dense and name a neighbouring page rather than
# expand this one. Advisory only — no gate, no rejection.
_DENSITY_NUDGE_CHARS = 30000


def _body_block_or_stub(mode: str, wiki_id: str | None, old_body: str) -> str:
    """For attach mode with a body too large to safely inline, return a
    stub directing the agent to use the section tools instead. Small
    bodies and other modes inline as before."""
    if not old_body:
        return "(none — create mode)"
    if mode == "attach" and wiki_id and len(old_body) > _INLINE_BODY_MAX_CHARS:
        nudge = ""
        if len(old_body) > _DENSITY_NUDGE_CHARS:
            nudge = (
                f"\n\nThis page is ALREADY LONG ({len(old_body)} chars). Prefer\n"
                f"density over growth: integrate into existing prose rather than\n"
                f"append parallel sentences, and where a NON-member detail you\n"
                f"turned up belongs to a neighbouring subject, name that page in\n"
                f"prose instead of restating it here — or leave it out; it stays\n"
                f"an orphan and comes back for its own page later. This never\n"
                f"applies to a MEMBER of this job: every member must be cited\n"
                f"here regardless of length."
            )
        return (
            f"[BODY OMITTED — {len(old_body)} chars, too large to inline.\n"
            f"Use the section tools to navigate without consuming context:\n"
            f"  - read_wiki_outline(\"{wiki_id}\") — section list + sizes + revision\n"
            f"  - read_wiki_section(\"{wiki_id}\", \"<section_name>\") — one section\n"
            f"  - edit_wiki_section(...) per section, validate_wiki, then\n"
            f"    final_answer(mode=\"attach\", body=\"\") — router persists via\n"
            f"    section edits and skips the full-body write.]"
            f"{nudge}"
        )
    return old_body


@router.post("/write")
async def wiki_write():
    """
    Write/update ONE wiki (one target per call). The LLM authors the entire
    body and may freely revise summary/disambiguation/scope/any section. No
    content gate, no manifest, no code-built ledger. The only guarantees are
    process/bookkeeping: the prior version is snapshotted (reversible) and
    `summarises` relations are reconciled *additively* from the LLM's inline
    refs. The LLM researches with its own tools before writing.
    """
    # 1. Pick + claim a bucket.
    with get_conn() as conn:
        bucket = wiki_jobs.next_write_bucket(conn)
        if not bucket:
            return {"written": 0, "message": "no pending create/attach jobs"}
        mode = bucket["mode"]
        jobs = bucket["jobs"]
        job_ids = [str(j["id"]) for j in jobs]
        lock_key = bucket["target_wiki_id"] or f"create:{job_ids[0]}"
        if not wiki_jobs.try_wiki_lock(conn, lock_key):
            return {"written": 0, "message": "target locked by another writer; retry later"}
        claimed = wiki_jobs.claim_jobs(conn, job_ids)
        if not claimed:
            return {"written": 0, "message": "jobs no longer claimable"}

        member_ids: list[str] = []
        for j in jobs:
            member_ids.extend(j["entity_ids"])
        dupes: list[dict] = []
        if mode == "attach":
            members = wiki_jobs.fetch_members(conn, member_ids)
            wiki = wiki_jobs.fetch_wiki(conn, bucket["target_wiki_id"])
            if not wiki:
                wiki_jobs.finish_jobs(conn, job_ids, "failed", "target wiki missing")
                return {"written": 0, "result": "failed", "reason": "target wiki missing"}
            canonical = wiki["canonical_name"]
            old_body = wiki["content"] or ""
            # Snapshot at CLAIM, not at persist. The snapshot's CONTENT was
            # always this claim-time body — only its WRITE used to happen at
            # persist, which meant a run that died after a section edit left
            # a committed mutation with no snapshot anywhere. Writing it here
            # (same transaction as the claim) closes that window; the section
            # tools' own commits are now always preceded by a durable
            # "this run started from revision N" record. A run that later
            # fails or no-ops leaves this row too — truthful, and bounded by
            # max_attempts.
            wiki_jobs.snapshot_revision(
                conn, str(bucket["target_wiki_id"]), old_body,
                wiki_jobs.parse_refs(old_body), wiki["revision"])
        elif mode == "consolidate":
            members = []
            dupes = wiki_jobs.fetch_wikis_for_merge(conn, bucket["wiki_ids"])
            if len(dupes) < 2:
                wiki_jobs.finish_jobs(conn, job_ids, "failed",
                                      "fewer than 2 live wikis to consolidate")
                return {"written": 0, "result": "failed", "reason": "nothing to merge"}
            canonical = "(decide among duplicates)"
            wiki = None
            old_body = "\n\n".join(d["content"] or "" for d in dupes)
            # Same claim-time snapshot rule as attach, one per duplicate
            # (this now precedes the canonical_no validation at persist —
            # a snapshot of a merge that then failed validation is a
            # harmless, truthful record).
            for d in dupes:
                wiki_jobs.snapshot_revision(
                    conn, d["id"], d["content"] or "",
                    wiki_jobs.parse_refs(d["content"] or ""), d["revision"])
        else:  # create
            members = wiki_jobs.fetch_members(conn, member_ids)
            canonical = bucket["proposed_name"] or "Untitled"
            wiki = None
            old_body = ""
        batch_id = str(jobs[0].get("batch_id")) if jobs[0].get("batch_id") else None
        # Neighbouring pages the writer may name in prose instead of expanding
        # this one. Short and relevant by construction — NOT the full catalog.
        # Never in consolidate mode: there `member_ids` holds the DUPLICATE
        # WIKI ids, so this would hand the writer the very pages it must
        # absorb and invite it to link out to them instead of merging them.
        related = (
            []
            if mode == "consolidate"
            else wiki_jobs.list_related_wikis(
                conn, member_ids, bucket.get("target_wiki_id")
            )
        )

    def _related_wikis_block(rows: list[dict]) -> str:
        if not rows:
            return "(none — no neighbouring pages exist yet)"
        return "\n".join(
            f"- {r['canonical_name']} ({r['char_count']}ch)" for r in rows
        )

    def _dupes_block(ds: list[dict]) -> str:
        if not ds:
            return "(n/a)"
        # Numbered; the writer picks the survivor by NUMBER (canonical_no),
        # never by id. This order IS the numbering resolved below.
        return "\n".join(
            f"{i}. {d['canonical_name']} "
            f"(importance: {d['importance']}  revision: {d['revision']})\n"
            f"  body:\n{(d['content'] or '')[:3000]}"
            for i, d in enumerate(ds, 1)
        )

    # 2. One focused agent call. Optional custom-profile shaping: a
    # `.replace.md` may swap the template (it then owns the `%%...%%` tokens);
    # `.add.md` fragments are appended after substitution. No active profile ->
    # both are no-ops and this prompt is byte-identical to the base.
    _writer_tmpl = custom_profile.replace_or_base(_WRITER_PROMPT, "wiki_writer")
    prompt = (
        _writer_tmpl
        .replace("%%MODE%%", mode)
        .replace("%%CANONICAL%%", canonical)
        .replace("%%WIKI_ID%%", bucket["target_wiki_id"] or "(assigned after write)")
        .replace("%%MEMBERS%%", _members_block(members))
        .replace("%%CURRENT_BODY%%", _body_block_or_stub(mode, bucket.get("target_wiki_id"), old_body))
        .replace("%%RELATED_WIKIS%%", _related_wikis_block(related))
        .replace("%%DUPLICATES%%", _dupes_block(dupes))
    ) + _now_line() + custom_profile.additions("wiki_writer")
    # Capture pre-run revision on the target wiki for `attach` mode so we
    # can detect whether the writer used the section-edit tools (each
    # bumps `wikis_ext.revision` directly). The writer may then submit an
    # empty `body` — section edits are the authoritative persistence
    # path in that case. `create`/`consolidate` modes don't have a
    # pre-determined target, so empty body is rejected there.
    pre_revision: int | None = None
    if mode == "attach" and bucket.get("target_wiki_id"):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT revision FROM wikis_ext WHERE entity_id = %s::uuid",
                    (bucket["target_wiki_id"],),
                )
                row = cur.fetchone()
                if row:
                    pre_revision = row[0]

    # Generous turns so the writer can recall_memory / view_tree / delegate a
    # subagent to research and verify before writing.
    # `run_typed` returns a SDK-validated WikiWriteResult, or raises if the
    # model never submitted — handled below like any agent failure
    # (release + log + 5xx). The only extra guard is "non-empty body OR
    # section edits happened"; everything else is the model's job (and
    # validated by Pydantic).
    #
    # Context-handoff loop: the writer may end early via
    # `handoff_to_successor` when its context approaches the limit (see
    # `braindb/agent/hooks.py` token-budget watch + `tools.py` handoff
    # tool). We install a per-run handoff slot, run the agent, and if
    # the slot was filled we spawn a successor agent — same prompt, same
    # tools, fresh context — seeded with the previous agent's brief.
    # Bounded by `agent_writer_handoff_max_depth` so a misbehaving model
    # cannot recurse forever.
    handoff_slot, handoff_token = install_handoff_slot()
    try:
        try:
            res: WikiWriteResult = await run_typed(
                prompt, get_writer_agent(), WikiWriteResult, max_turns=30,
                token_budget=settings.agent_writer_handoff_token_budget,
            )
            depth = 0
            max_depth = settings.agent_writer_handoff_max_depth
            while handoff_slot.captured and depth < max_depth:
                depth += 1
                # Appended AFTER the rendered job prompt (see run_typed call
                # below), so the successor really does have the same prompt:
                # mode, canonical name, wiki id, MEMBERS, the body stub, the
                # neighbouring pages and every writer rule. Seeding it with
                # this brief ALONE — as this loop used to — left it running on
                # the generic SYSTEM_PROMPT with no members, no preservation
                # rule and no `body=""` contract, so it re-read from scratch
                # exactly what the brief had just told it.
                seed = (
                    "---\n\n"
                    "CONTINUATION. The job above is unchanged. A previous "
                    "agent run on it ended early via `handoff_to_successor` "
                    "because its context was filling up. You have its brief "
                    "below, the same tools, and a fresh context window.\n\n"
                    "PROGRESS SO FAR (from the previous agent):\n"
                    f"{handoff_slot.progress_summary}\n\n"
                    "REMAINING WORK:\n"
                    f"{handoff_slot.remaining_work}\n\n"
                    "Trust the brief for STATE — revisions, decisions made, "
                    "sections already done — and do not re-derive those. But "
                    "your fresh context does not hold any section's TEXT: "
                    "still read any section (or the header) you intend to "
                    "replace, and choose edits by content, not by cost, "
                    "exactly as the job instructions above say. Call "
                    "`final_answer` when done — body=\"\" is for ATTACH mode "
                    "only (when your section edits persisted the content); "
                    "create and consolidate must submit the full body. If "
                    "YOUR context also fills up before you finish, call "
                    "`handoff_to_successor` again with an updated brief — "
                    "the same successor mechanism will continue."
                )
                handoff_slot.captured = False
                handoff_slot.progress_summary = ""
                handoff_slot.remaining_work = ""
                logger.info(
                    "writer handoff: spawning successor #%d/%d (mode=%s, jobs=%s)",
                    depth, max_depth, mode, job_ids,
                )
                res = await run_typed(
                    f"{prompt}\n\n{seed}", get_writer_agent(), WikiWriteResult,
                    max_turns=30,
                    token_budget=settings.agent_writer_handoff_token_budget,
                )
            if handoff_slot.captured:
                # Depth cap hit AND last run still asked for handoff —
                # treat as a failure (the model isn't converging).
                logger.warning(
                    "writer handoff: depth cap %d hit; treating as failure",
                    max_depth,
                )
                with get_conn() as conn:
                    disp = wiki_jobs.release_or_fail_jobs(
                        conn, job_ids,
                        f"handoff depth cap {max_depth} exhausted "
                        f"without final_answer",
                    )
                    log_activity(conn, "wiki_write", "wiki",
                                 bucket.get("target_wiki_id"), details={
                                     "result": disp, "mode": mode,
                                     "jobs": job_ids,
                                     "error": "handoff depth exhausted",
                                 })
                return {"written": 0, "result": disp,
                        "reason": "handoff depth exhausted"}
        except Exception as e:
            logger.exception("writer agent failed")
            with get_conn() as conn:
                disp = wiki_jobs.release_or_fail_jobs(conn, job_ids, f"agent error: {e}")
                # Mirror the success-path wiki_write row so a DB-only
                # auditor sees crashed runs, not a suspiciously clean log.
                log_activity(conn, "wiki_write", "wiki",
                             bucket.get("target_wiki_id"), details={
                                 "result": disp, "mode": mode,
                                 "jobs": job_ids, "error": str(e)[:500],
                             })
            return {"written": 0, "result": disp, "reason": str(e)}
    finally:
        release_handoff_slot(handoff_token)

    if _is_blank_body(res.body):
        # Empty body — only valid in attach mode if section edits bumped
        # the revision during the run. Otherwise the agent did nothing
        # persistable and we fail the jobs.
        if mode != "attach" or pre_revision is None:
            with get_conn() as conn:
                disp = wiki_jobs.release_or_fail_jobs(
                    conn, job_ids,
                    f"empty body returned in {mode} mode: "
                    f"{res.model_dump_json()[:300]}",
                )
            return {"written": 0, "result": disp, "reason": "no body returned"}
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT e.content, w.revision
                       FROM entities e JOIN wikis_ext w ON w.entity_id = e.id
                       WHERE e.id = %s::uuid""",
                    (bucket["target_wiki_id"],),
                )
                row = cur.fetchone()
        if not row:
            with get_conn() as conn:
                disp = wiki_jobs.release_or_fail_jobs(
                    conn, job_ids,
                    "empty body AND wiki vanished",
                )
            return {"written": 0, "result": disp, "reason": "wiki missing"}
        if row[1] == pre_revision:
            # No section edits happened. This is legitimate ONLY if every
            # assigned member is already cited in the body — then there is
            # nothing to write and we just need to run reconcile so the
            # `summarises` relations catch up. If any member is missing
            # from the body, the writer skipped real work — fail it.
            body_now = row[0] or ""
            with get_conn() as conn:
                # The shared predicate (`uncited_members`) splits un-cited
                # members into `missing` (entity exists — real outstanding
                # work) and `gone` (entity deleted since triage — can never
                # be cited, and must not wedge the job in a retry loop; the
                # same premise as reconcile's dangling-ref skip).
                missing, gone = wiki_jobs.uncited_members(
                    conn, body_now, member_ids)
                if missing:
                    disp = wiki_jobs.release_or_fail_jobs(
                        conn, job_ids,
                        f"empty body AND no section edits AND "
                        f"{len(missing)} member(s) not yet cited in body",
                    )
                    return {"written": 0, "result": disp,
                            "reason": "members un-cited"}
                # Every existing member cited — close the no-op cleanly
                # and reconcile. `gone` ids are recorded, never silent.
                rel = wiki_jobs.reconcile_summarises_additive(
                    conn, bucket["target_wiki_id"], body_now)
                wiki_jobs.finish_jobs(conn, job_ids, "done")
                log_activity(conn, "wiki_write", "wiki",
                             bucket["target_wiki_id"], details={
                                 "mode": mode, "no_op": True,
                                 "revision": pre_revision,
                                 "members": len(member_ids),
                                 "members_gone": gone, **rel,
                             })
            logger.info(
                "writer no-op accepted: pre_rev=%s, all %d members already "
                "cited (%d gone); reconcile=%s",
                pre_revision, len(member_ids), len(gone), rel,
            )
            return {"written": 0, "wiki_id": bucket["target_wiki_id"],
                    "mode": mode, "revision": pre_revision,
                    "jobs": job_ids, "no_op": True, **rel}
        new_body = row[0]
        logger.info(
            "writer used section-edit path: pre_rev=%s post_rev=%s body=%dch",
            pre_revision, row[1], len(new_body),
        )
    else:
        new_body = res.body

    # Final blank-body backstop. `new_body` is now whatever we are about to
    # persist — either the LLM's submission or the post-section-edit DB body.
    # A genuine page always has real characters; only serialization garbage
    # (`""`, `body=""`, `{}`) reaches here blank. Persisting it would write an
    # empty page AND bump the revision, which self-perpetuates into a high-rev
    # empty loop. Reject instead (same release/retry path as the gates above) —
    # the entity stays a healable orphan and is re-tried next cycle. The legit
    # no-op (all members already cited) already returned above, so this never
    # touches it.
    if _is_blank_body(new_body):
        with get_conn() as conn:
            disp = wiki_jobs.release_or_fail_jobs(
                conn, job_ids,
                f"blank/garbage body not persisted ({mode}): {new_body[:80]!r}")
        return {"written": 0, "result": disp, "reason": "blank body"}

    # 3. Persist (one transaction). The LLM's body is authoritative (guarded
    #    above against blank); we only snapshot (reversible) and reconcile
    #    additively.
    with get_conn() as conn:
        summary, disambig = wiki_jobs.extract_summary_disambig(new_body)
        kw = wiki_jobs.keywords_from_meta(new_body)
        retired: list[str] = []
        if mode == "create":
            wiki_id = wiki_jobs.create_wiki_entity(
                conn, canonical, new_body, summary, disambig, member_ids,
                keywords=kw)
            revision = 1
        elif mode == "consolidate":
            no = res.canonical_no
            if not (isinstance(no, int) and 1 <= no <= len(dupes)):
                disp = wiki_jobs.release_or_fail_jobs(
                    conn, job_ids,
                    f"canonical_no {no!r} not a valid duplicates number (1..{len(dupes)})")
                return {"written": 0, "result": disp,
                        "reason": "invalid canonical_no"}
            canonical_id = dupes[no - 1]["id"]
            wiki_id = canonical_id
            # Snapshots were written at CLAIM time (see the claim block) —
            # every duplicate's pre-run body is already durably recorded.
            revision = wiki_jobs.finalize_wiki_write(
                conn, wiki_id, new_body, summary, disambig, member_ids)
            for d in dupes:
                if d["id"] != canonical_id:
                    wiki_jobs.soft_retire_wiki(conn, d["id"], canonical_id, None)
                    retired.append(d["id"])
        else:  # attach
            wiki_id = bucket["target_wiki_id"]
            # Snapshot was written at CLAIM time (see the claim block).
            revision = wiki_jobs.finalize_wiki_write(
                conn, wiki_id, new_body, summary, disambig, member_ids)

        rel = wiki_jobs.reconcile_summarises_additive(conn, wiki_id, new_body)

        wiki_jobs.finish_jobs(conn, job_ids, "done")
        log_activity(conn, "wiki_write", "wiki", wiki_id, details={
            "mode": mode, "revision": revision, "jobs": job_ids,
            "members": len(member_ids), "retired": retired, **rel,
        })

    return {"written": 1, "wiki_id": wiki_id, "mode": mode,
            "revision": revision, "jobs": job_ids, "retired": retired, **rel}


def _is_wiki(conn, entity_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM entities WHERE id = %s AND entity_type = 'wiki'", (str(entity_id),))
        return cur.fetchone() is not None


@router.get("/jobs")
def wiki_jobs_list(
    status: str | None = Query(default=None),
    job_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    with get_conn() as conn:
        return wiki_jobs.list_jobs(conn, status, job_type, limit)
