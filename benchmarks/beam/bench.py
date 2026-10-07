"""BrainDB on BEAM bench runner.

Per-conversation lifecycle (zero interleaving within a conversation):

    Phase A — create new Postgres database `braindb_conv_NNN`, restart
              api_bench with DATABASE_URL pointing at it, wait for healthy,
              write conversation .md to data_bench/sources/
    Phase B — warmup wait (extraction settles; wikis run async in background)
    Phase C — answer this conversation's 20 probing questions via /agent/query
    Phase D — record answers + probing_questions + warmup_stats + meta to
              runs/<run_id>/conv_<NNN>/

After all conversations: 20 (or 35) databases sit in postgres_bench, each
fully self-contained and inspectable (`docker exec braindb_bench_postgres
psql -U braindb_bench -d braindb_conv_007 -c '\\dt'`).

Strict safety: every destructive op (CREATE DATABASE, DROP DATABASE, etc.)
is gated on the literal substring `braindb_bench` OR `braindb_conv_`
appearing in the URL. The personal `braindb` database is never touched.

CLI:

    python -m benchmarks.beam.bench --split 100K --limit 1       # smoke (1 conv)
    python -m benchmarks.beam.bench --split 100K                  # full 100K (20 convs)
    python -m benchmarks.beam.bench --split 1M --limit 3 --fail-fast

Intended to run from inside the `bench_runner` container (which has the
docker CLI installed + docker socket mounted so it can recreate api_bench).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import psycopg2
import requests

from benchmarks.beam.adapter import (
    Conversation,
    iter_conversations,
    load_probing_questions,
    write_conversation_md,
)
from benchmarks.beam.config import (
    BENCH_API_BASE,
    BENCH_DATABASE_URL,
    DATA_BENCH_SOURCES,
    RUNS_DIR,
    assert_bench_database_url,
)
from benchmarks.beam.warmup import wait_for_warmup

# Admin URL + DB-URL base for per-conversation database creation. The
# bench_runner service in docker-compose.bench.yml supplies these via env;
# fall back to sensible defaults that match the bench network.
_ADMIN_DB_URL = os.getenv(
    "BENCH_ADMIN_DATABASE_URL",
    "postgresql://braindb_bench:bench_local_only@postgres_bench:5432/postgres",
)
_DB_BASE_URL = os.getenv(
    "BENCH_DB_BASE_URL",
    "postgresql://braindb_bench:bench_local_only@postgres_bench:5432",
)

# docker-compose file path inside the bench_runner container. The compose
# file is mounted via the repo bind mount; `COMPOSE_FILE` env var hints
# the same path so `docker compose ...` picks it up automatically.
_COMPOSE_FILE = os.getenv("COMPOSE_FILE", "/app/docker-compose.bench.yml")

# Per-question agent timeout. BrainDB's /agent/query can take a while on
# local Qwen — multi-turn agent loops + extraction can be 30s to several
# minutes per question. 10 minutes hard ceiling per question.
QUESTION_TIMEOUT_SECONDS = 600


# ----------------------------- helpers ---------------------------------------

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
        )
        return bool(out.strip())
    except Exception:
        return False


# ----------------------------- preflight -------------------------------------

def check_bench_api_healthy(base: str = BENCH_API_BASE, timeout: float = 5) -> None:
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        raise RuntimeError(
            f"bench API not reachable at {base} ({e}). "
            f"Did you run: docker compose -f docker-compose.bench.yml up -d ?"
        )
    if body.get("status") != "ok":
        raise RuntimeError(f"bench API unhealthy: {body}")


# ----------------------------- per-conv DB orchestration ---------------------

def _conv_db_name(conv_id: int) -> str:
    """Return the database name for a given conversation_id (zero-padded)."""
    return f"braindb_conv_{int(conv_id):03d}"


def _conv_db_url(conv_id: int) -> str:
    """Full DATABASE_URL for the conversation's database."""
    return f"{_DB_BASE_URL}/{_conv_db_name(conv_id)}"


def create_conv_db(conv_id: int) -> str:
    """Create a fresh Postgres database `braindb_conv_NNN`.

    If a database with the same name exists from a prior run, drop + recreate
    it (the per-conv DB is supposed to be a clean slate). Returns the full
    connection URL for the new database.

    Safety: hits postgres_bench via the admin connection (`postgres` maintenance
    DB) — never the user's personal `braindb`.
    """
    db_name = _conv_db_name(conv_id)
    db_url = _conv_db_url(conv_id)
    assert_bench_database_url(db_url)
    admin = psycopg2.connect(_ADMIN_DB_URL)
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            # Terminate any leftover backend on this DB before drop (otherwise
            # DROP fails with "is being accessed by other users").
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
            cur.execute(f'CREATE DATABASE "{db_name}" OWNER braindb_bench')
    finally:
        admin.close()
    return db_url


def restart_api_with_db(database_url: str) -> None:
    """Recreate the api_bench container with DATABASE_URL set to ``database_url``.

    We use the Docker Python SDK rather than ``docker compose`` because bench.py
    runs inside the bench_runner container. ``docker compose -f .../docker-compose.bench.yml
    up`` would resolve the compose file's relative paths (e.g. ``.:/app``) against
    the runner's view of the filesystem, not the host's. The recreated api_bench
    would then bind-mount the runner's /app (which is the workstation host dir,
    but only because of OUR bind mount — fragile and wrong from the daemon's POV).

    The SDK path captures the existing api_bench container's config (image,
    binds, networks, ports, etc.) and recreates it with that config plus the
    updated ``DATABASE_URL`` env var.
    """
    assert_bench_database_url(database_url)
    import docker as docker_sdk  # imported lazily so the module loads on hosts
                                  # that don't have the SDK installed
    client = docker_sdk.from_env()
    try:
        existing = client.containers.get("braindb_bench_api")
    except docker_sdk.errors.NotFound:  # type: ignore[attr-defined]
        raise RuntimeError(
            "braindb_bench_api container not found — bring the compose stack "
            "up first: `docker compose -f docker-compose.bench.yml up -d`"
        )

    cfg = existing.attrs
    image = cfg["Config"]["Image"]
    host_cfg = cfg["HostConfig"]
    net_settings = cfg["NetworkSettings"]

    # Preserve existing volumes (host paths the daemon already resolved when
    # the container was first created via `docker compose up`).
    binds = list(host_cfg.get("Binds") or [])

    # Networks: capture name + aliases from each. Compose sets the SERVICE
    # NAME (`api_bench`) as a network alias so other services can DNS-resolve
    # it; the SDK doesn't auto-restore that alias on container recreate, so
    # we must replay it explicitly. We always include "api_bench" as a
    # fallback alias even if the existing container's aliases came back
    # empty (which docker inspect does sometimes report).
    networks_to_connect = {}
    for net_name, net_cfg in (net_settings.get("Networks") or {}).items():
        aliases = set(net_cfg.get("Aliases") or [])
        aliases.add("api_bench")
        networks_to_connect[net_name] = sorted(aliases)

    # Ports: preserve the existing host port mapping.
    port_bindings = {}
    for port_proto, mappings in (host_cfg.get("PortBindings") or {}).items():
        if mappings:
            port_bindings[port_proto] = mappings[0]["HostPort"]

    # Env: replace DATABASE_URL (or add if missing).
    env_list = list(cfg["Config"].get("Env") or [])
    new_env = [e for e in env_list if not e.startswith("DATABASE_URL=")]
    new_env.append(f"DATABASE_URL={database_url}")

    # Stop + remove existing.
    existing.stop(timeout=30)
    existing.remove()

    # Recreate with same shape + new env. Use create()+start() instead of
    # run() so we can connect networks with aliases BEFORE the container
    # boots (run(network=...) does not accept aliases). Without the alias
    # replay, the recreated container loses its `api_bench` DNS name and
    # bench_runner (which talks to it via `http://api_bench:8001`) fails
    # to resolve it.
    container = client.containers.create(
        image,
        name="braindb_bench_api",
        environment=new_env,
        volumes=binds,
        ports=port_bindings,
        extra_hosts={
            h.split(":", 1)[0]: h.split(":", 1)[1]
            for h in (host_cfg.get("ExtraHosts") or [])
        } or None,
        restart_policy=(
            {"Name": host_cfg["RestartPolicy"]["Name"]}
            if host_cfg.get("RestartPolicy") and host_cfg["RestartPolicy"].get("Name")
            else None
        ),
        command=cfg["Config"].get("Cmd"),
        working_dir=cfg["Config"].get("WorkingDir") or None,
    )
    # Disconnect from the default bridge that `create` joined the container to,
    # then attach the captured user-defined networks with their aliases.
    try:
        client.networks.get("bridge").disconnect(container, force=True)
    except docker_sdk.errors.APIError:  # type: ignore[attr-defined]
        pass  # not on the bridge — fine
    for net_name, aliases in networks_to_connect.items():
        try:
            client.networks.get(net_name).connect(container, aliases=aliases)
        except docker_sdk.errors.NotFound:  # type: ignore[attr-defined]
            pass
    container.start()


def wait_for_api_healthy(timeout: float = 300, poll: float = 2) -> None:
    """Block until the bench api responds with status=ok.

    The api needs to (1) connect to the new DB, (2) run alembic migrations,
    (3) load the embedding model into memory. On first start this is ~30-60s;
    on subsequent restarts (model already cached) it can still be ~20-30s.
    Bumped 180s -> 300s for generous slack on cold-cache restarts and to
    absorb an occasional slow alembic migration without tripping.
    """
    start = time.monotonic()
    last_err = None
    while time.monotonic() - start < timeout:
        try:
            r = requests.get(f"{BENCH_API_BASE}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(poll)
    raise TimeoutError(
        f"bench api did not become healthy within {timeout:.0f}s (last: {last_err})"
    )


def _clear_sources_dir() -> None:
    """Remove any stale .md files (and their .error.txt sidecars) left in
    the bench watcher's source dir and its ingested/ + failed/ siblings.
    Without this, files from previous conversations would either be
    re-ingested by the next warmup OR poison the warmup barrier by leaving
    pending content in the watch dir.
    """
    DATA_BENCH_SOURCES.mkdir(parents=True, exist_ok=True)
    for f in DATA_BENCH_SOURCES.glob("*.md"):
        try:
            f.unlink()
        except OSError:
            pass
    # The watcher creates ingested/ + failed/ as siblings of WATCH_DIR
    # (see ingest_watcher.py lines 39-40). Clean both so cumulative runs
    # don't pile up on disk and so a previous failed file doesn't trick
    # the next run into thinking ingest already happened.
    for sibling in ("ingested", "failed"):
        d = DATA_BENCH_SOURCES / sibling
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except OSError:
                        pass


# ----------------------------- ask agent -------------------------------------

def answer_one_question(
    question_text: str,
    base: str = BENCH_API_BASE,
    timeout: float = QUESTION_TIMEOUT_SECONDS,
    need: str | None = None,
) -> tuple[str, dict]:
    """POST /agent/query with the question; return (answer_text, raw_payload).

    `need`, when given, is sent as the request's `need` (JEV runs).

    Three-attempt retry with 5s/10s backoff. Retries on connection errors,
    timeouts, and HTTP 5xx — these are the transient classes (API recycling,
    LLM backend blip, vLLM batching pause). Does NOT retry on HTTP 4xx —
    those are request-shape errors and will repeat. After 3 failed attempts
    the last exception is raised so the caller records `[ERROR: ...]` and
    the run continues to the next question.
    """
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        started = time.monotonic()
        try:
            r = requests.post(
                f"{base}/api/v1/agent/query",
                json={"query": question_text, **({"need": need} if need else {})},
                timeout=timeout,
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
        else:
            if 500 <= r.status_code < 600:
                last_exc = requests.HTTPError(
                    f"HTTP {r.status_code} on attempt {attempt}/3: {r.text[:200]}",
                    response=r,
                )
            else:
                r.raise_for_status()  # 4xx -> raise immediately, caller records
                elapsed = time.monotonic() - started
                payload = r.json()
                payload["_elapsed_seconds"] = round(elapsed, 2)
                if attempt > 1:
                    payload["_attempts"] = attempt
                # BrainDB's /agent/query response shape: {"answer": "..."} at minimum.
                return payload.get("answer", ""), payload
        # Got a transient error — back off and retry unless this was the last try
        if attempt < 3:
            time.sleep(5 * attempt)   # 5s, 10s
    assert last_exc is not None
    raise last_exc


# ----------------------------- per-conv lifecycle ----------------------------

def run_one_conversation(
    conv: Conversation,
    run_dir: Path,
    *,
    warmup_timeout: float = 43200,
    warmup_settle_seconds: float = 600,
    question_timeout: float = QUESTION_TIMEOUT_SECONDS,
    block_on_wiki_queue: bool = False,
    wiki_wait_seconds: float = 0,
) -> dict:
    """Full A-B-C-D lifecycle for one conversation. Returns a small stats dict."""
    conv_dir = run_dir / f"conv_{int(conv.conversation_id):03d}"
    conv_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    conv_db_name = _conv_db_name(int(conv.conversation_id))
    conv_db_url = _conv_db_url(int(conv.conversation_id))

    # ---- Phase A: fresh DB + restart api + ingest ----
    print(f"[A] create DB '{conv_db_name}' + restart api + write conversation ...",
          flush=True)
    create_conv_db(int(conv.conversation_id))
    restart_api_with_db(conv_db_url)
    wait_for_api_healthy()
    _clear_sources_dir()
    md_path = write_conversation_md(conv, DATA_BENCH_SOURCES)
    md_kb = md_path.stat().st_size / 1024
    print(
        f"[A done] api restarted on {conv_db_name}; "
        f"{md_path.name} ({md_kb:.0f} KB) dropped at {_now_iso()}",
        flush=True,
    )

    # ---- Phase B: warmup wait ----
    print(f"[B] warmup wait ...", flush=True)
    warmup_stats = wait_for_warmup(
        settle_seconds=warmup_settle_seconds,
        timeout_seconds=warmup_timeout,
        verbose=True,
        block_on_wiki_queue=block_on_wiki_queue,
        database_url=conv_db_url,
    )
    print(f"[B done] {warmup_stats}", flush=True)

    # ---- Phase Bw (optional): wait for wiki writer to drain its queue ----
    # Between [B done] (extraction settled) and [C] (questions start), give
    # the wiki writer/maintainer dedicated vLLM time so wikis develop before
    # recall queries hit. Skipped when wiki_wait_seconds == 0 (default).
    if wiki_wait_seconds and wiki_wait_seconds > 0:
        print(
            f"[Bw] waiting {wiki_wait_seconds:.0f}s for wiki queue to develop ...",
            flush=True,
        )
        time.sleep(wiki_wait_seconds)
        print(f"[Bw done]", flush=True)

    # ---- Phase C: answer all questions ----
    probing = load_probing_questions(conv)
    print(f"[C] answering {sum(len(v) for v in probing.values())} questions "
          f"across {len(probing)} categories ...", flush=True)
    answers_by_category: dict[str, list[dict]] = {}
    raw_payloads_by_category: dict[str, list[dict]] = {}
    question_errors = 0

    for category, questions in probing.items():
        category_answers: list[dict] = []
        category_payloads: list[dict] = []
        for i, q in enumerate(questions, start=1):
            question_text = q["question"]
            try:
                ans, payload = answer_one_question(
                    question_text, timeout=question_timeout
                )
                print(
                    f"  [{category} {i}/{len(questions)}] OK "
                    f"({payload.get('_elapsed_seconds', '?')}s, "
                    f"{len(ans)} chars)",
                    flush=True,
                )
            except Exception as e:
                ans = f"[ERROR: {type(e).__name__}: {e}]"
                payload = {"_error": str(e)}
                question_errors += 1
                print(f"  [{category} {i}/{len(questions)}] ERROR: {e}", flush=True)
            category_answers.append({"question": question_text, "llm_response": ans})
            category_payloads.append({"question": question_text, "payload": payload})
        answers_by_category[category] = category_answers
        raw_payloads_by_category[category] = category_payloads

    # ---- Phase D: record artefacts ----
    print(f"[D] recording artefacts to {conv_dir} ...", flush=True)
    conv_dir.joinpath("answers.json").write_text(
        json.dumps(answers_by_category, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    conv_dir.joinpath("probing_questions.json").write_text(
        json.dumps(probing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    conv_dir.joinpath("warmup_stats.json").write_text(
        json.dumps(warmup_stats, indent=2), encoding="utf-8"
    )
    conv_dir.joinpath("raw_payloads.json").write_text(
        json.dumps(raw_payloads_by_category, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    conv_meta = {
        "conversation_id": conv.conversation_id,
        "split": conv.split,
        "slug": conv.slug,
        "category": (conv.raw.get("conversation_seed") or {}).get("category"),
        "ingested_md_size_kb": int(md_kb),
        "question_errors": question_errors,
        "wall_clock_s": round(time.monotonic() - started, 1),
        "started_at": _now_iso(),
        "bench_db_name": conv_db_name,
    }
    conv_dir.joinpath("conversation_meta.json").write_text(
        json.dumps(conv_meta, indent=2), encoding="utf-8"
    )

    print(f"[done] conv {conv.slug} in {conv_meta['wall_clock_s']:.0f}s, "
          f"errors={question_errors}\n", flush=True)
    return conv_meta


# ----------------------------- run-level -------------------------------------

def _generate_run_id(split: str, limit: int | None) -> str:
    sha = _git_sha()
    dirty = "+dirty" if _git_dirty() else ""
    n = f"_n{limit}" if limit is not None else ""
    return f"{_short_ts()}_{split}{n}_{sha}{dirty}"


def _save_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    cfg = {
        "split": args.split,
        "limit": args.limit,
        "fail_fast": args.fail_fast,
        "warmup_timeout": args.warmup_timeout,
        "warmup_settle_seconds": args.warmup_settle_seconds,
        "question_timeout": args.question_timeout,
        "started_at": _now_iso(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "bench_api_base": BENCH_API_BASE,
        "python_version": sys.version,
    }
    run_dir.joinpath("run_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )


def _conversations_to_run(args: argparse.Namespace) -> Iterable[Conversation]:
    convs = list(iter_conversations(args.split))
    if args.limit is not None:
        convs = convs[: args.limit]
    if args.only_ids:
        wanted = set(s.strip() for s in args.only_ids.split(","))
        convs = [c for c in convs if c.conversation_id in wanted]
    return convs


# ----------------------------- CLI -------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", default="1M", choices=["100K", "500K", "1M"])
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N conversations from the split")
    p.add_argument("--only-ids", default=None,
                   help="comma-separated list of conversation_ids to run (overrides --limit ordering)")
    p.add_argument("--fail-fast", action="store_true",
                   help="abort the whole run if any single conversation raises")
    p.add_argument("--warmup-timeout", type=float, default=43200,
                   help="seconds before warmup gives up on convergence (default 43200 = 12h). "
                        "Must exceed total per-conversation extraction time (a 100K conv takes "
                        "~5.7h at 1200-word chunks). settle_seconds=600s catches genuine stalls "
                        "inside this window so the full 12h is only burned in the "
                        "extraction-keeps-progressing case.")
    p.add_argument("--warmup-settle-seconds", type=float, default=600,
                   help="seconds of quiet on entities AND relations before declaring "
                        "warmup clear (default 600). Per-chunk agents have 2-4 min quiet "
                        "stretches doing subagent / recall_memory work between save_fact "
                        "and create_relation bursts; 10 min gives generous slack.")
    p.add_argument("--wait-for-wikis", action="store_true",
                   help="Strict warmup mode: also wait for the wiki_job queue to be "
                        "fully drained before answering. Off by default because the "
                        "wiki writer can be slower than the maintainer queues for "
                        "large documents, so the queue may never converge. Wikis "
                        "continue async in the background regardless.")
    p.add_argument("--question-timeout", type=float, default=QUESTION_TIMEOUT_SECONDS,
                   help="per-question HTTP timeout on /agent/query")
    p.add_argument("--wiki-wait-seconds", type=float, default=0,
                   help="seconds to wait after [B done] before starting Phase C, "
                        "so the wiki writer can drain its queue against an "
                        "otherwise-idle vLLM (default 0 = no wait). Recommended "
                        "value 28800 (8 hours) for BEAM 100K convs — captures most "
                        "of the queue-drain benefit; longer waits hit diminishing "
                        "returns because the maintainer keeps queuing new triage "
                        "jobs at roughly the rate the writer drains them.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    assert_bench_database_url()
    check_bench_api_healthy()

    convs = list(_conversations_to_run(args))
    if not convs:
        print("no conversations selected; exiting")
        return 1

    run_id = _generate_run_id(args.split, args.limit if not args.only_ids else None)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(run_dir, args)

    print(f"=== BEAM bench run: {run_id} ===")
    print(f"split={args.split}  conversations={len(convs)}  run_dir={run_dir}")
    print(f"bench API: {BENCH_API_BASE}")
    print(f"git SHA:   {_git_sha()}{'  (dirty)' if _git_dirty() else ''}")
    print()

    metas: list[dict] = []
    run_started = time.monotonic()
    for i, conv in enumerate(convs, start=1):
        print(f"--- Conversation {i}/{len(convs)} ({conv.slug}, "
              f"category={(conv.raw.get('conversation_seed') or {}).get('category')}) ---")
        try:
            meta = run_one_conversation(
                conv,
                run_dir,
                warmup_timeout=args.warmup_timeout,
                warmup_settle_seconds=args.warmup_settle_seconds,
                question_timeout=args.question_timeout,
                block_on_wiki_queue=args.wait_for_wikis,
                wiki_wait_seconds=args.wiki_wait_seconds,
            )
            metas.append(meta)
        except Exception as e:
            print(f"!!! conversation {conv.slug} failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            metas.append({
                "conversation_id": conv.conversation_id,
                "slug": conv.slug,
                "fatal_error": f"{type(e).__name__}: {e}",
            })
            if args.fail_fast:
                _write_run_summary(run_dir, metas, time.monotonic() - run_started)
                return 1

    _write_run_summary(run_dir, metas, time.monotonic() - run_started)
    print(f"\n=== run complete: {run_id} ===")
    return 0


def _write_run_summary(run_dir: Path, metas: list[dict], wall_clock_s: float) -> None:
    summary = {
        "finished_at": _now_iso(),
        "wall_clock_s": round(wall_clock_s, 1),
        "conversations_attempted": len(metas),
        "conversations_succeeded": sum(1 for m in metas if "fatal_error" not in m),
        "total_question_errors": sum(int(m.get("question_errors", 0)) for m in metas),
        "per_conv_meta": metas,
    }
    run_dir.joinpath("run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
