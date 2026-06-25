"""
Always-on wiki scheduler — ONE loop, like ingest_watcher.py (one interval).

Per tick:
  1. POST /wiki/cron               — cheap, pure SQL, no LLM.
  2. GET  /wiki/jobs?status=pending — cheap, pure SQL, no LLM. The gate.
  3. if a pending `triage` job exists  -> POST /wiki/maintain  (one case, C1)
  4. if pending suggestion jobs exist  -> POST /wiki/write, repeated to DRAIN
       them (bounded) so consolidate/attach keep up instead of trickling.
  5. nothing pending  -> NO LLM call this tick (idle == free).

The expensive LLM endpoints are never called speculatively: a tick with
empty queues costs nothing. No multi-timer staggering, one env var.
"""
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_URL = os.getenv("BRAINDB_API_URL", "http://localhost:8000")
INTERVAL = int(os.getenv("WIKI_INTERVAL", "60"))          # one cadence, like the watcher
DRAIN_MAX = int(os.getenv("WIKI_DRAIN_MAX", "20"))        # safety bound on /write per tick
# Per-tick concurrency: how many /wiki/write calls fire in parallel (vLLM
# continuous-batches them on the GPU; the DB layer is already safe via
# FOR UPDATE SKIP LOCKED on every claim and try_wiki_lock per wiki).
# `maintain` runs concurrently alongside writers (1 maintain in flight, C1).
WRITE_PARALLELISM = int(os.getenv("WIKI_WRITE_PARALLELISM", "3"))

# Master on/off for the whole wiki pipeline. Default ON — the wiki pipeline
# is part of what makes BrainDB a knowledge layer (vs. just a fact store),
# so a fresh stack starts the maintainer + writer automatically. Opt out
# explicitly with WIKI_ENABLED=false. Model-agnostic; orthogonal to any LLM
# profile/provider.
WIKI_ENABLED = os.getenv("WIKI_ENABLED", "true").lower() in ("1", "true", "yes", "on")
# HTTP read-timeout (seconds) the scheduler waits on a single /wiki/maintain
# or /wiki/write call before its requests client gives up and moves on.
# Bumped 600 → 1200 (10 → 20 min) after live observation on Qwen 27B AWQ-INT4
# (vLLM, workstation): full-body wiki writes routinely run 6-15 min on this
# model, so a 600s deadline at the scheduler caused the client to give up
# WHILE the api kept working in the background — the write still committed,
# but the scheduler couldn't see the completion in time to drain the queue
# efficiently. Bumped again 1200 → 2400 (20 → 40 min) for smaller/slower
# quantised models (e.g. Gemma 12B QAT-W4A16): their full-body writes can
# exceed 20 min, so a 1200s deadline abandoned slow-but-completing writes
# and forced needless retries. With 2400s the client waits out most writes
# while still surfacing genuinely-stuck jobs as failures rather than blocking
# indefinitely. The api itself is not bounded by this value; this knob only
# controls how patient the scheduler's HTTP client is.
AGENT_TIMEOUT = int(os.getenv("WIKI_AGENT_TIMEOUT", "2400"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [wiki-scheduler] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("wiki-scheduler")

_SUGGESTION_TYPES = {"create", "attach", "consolidate"}


def wait_for_api(timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    url = f"{API_URL}/health"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def _post(path: str, timeout: int) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
        log.warning("%s -> %s: %s", path, r.status_code, r.text[:200])
    except requests.RequestException as e:
        log.warning("%s request error: %s", path, e)
    return None


def _pending_kinds() -> tuple[bool, bool]:
    """(has_triage, has_suggestion) from ONE cheap SQL-only read. On error,
    return (False, False) so we never fire LLM calls on uncertain state."""
    try:
        r = requests.get(
            f"{API_URL}/api/v1/wiki/jobs",
            params={"status": "pending", "limit": 500},
            timeout=15,
        )
        if r.status_code != 200:
            log.warning("/jobs -> %s: %s", r.status_code, r.text[:200])
            return (False, False)
        jobs = r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("/jobs read error: %s", e)
        return (False, False)
    has_triage = any(j.get("job_type") == "triage" for j in jobs)
    has_sugg = any(j.get("job_type") in _SUGGESTION_TYPES for j in jobs)
    return (has_triage, has_sugg)


def main() -> None:
    if not WIKI_ENABLED:
        log.info("wiki pipeline DISABLED (set WIKI_ENABLED=true to enable). Idle.")
        # Sleep forever — keeps the container Up without restart-loop, and
        # makes zero LLM/DB/api calls. Toggle via env + scheduler restart.
        while True:
            time.sleep(3600)

    log.info("waiting for API at %s ...", API_URL)
    if not wait_for_api():
        log.error("API never came up; exiting")
        sys.exit(1)
    log.info("wiki scheduler ready (single loop, interval=%ss)", INTERVAL)

    while True:
        try:
            # 1. cron — cheap SQL, safe to run every tick.
            res = _post("/api/v1/wiki/cron", timeout=60)
            if res and res.get("triage_jobs_enqueued"):
                log.info("cron: enqueued=%s pending_triage=%s",
                         res.get("triage_jobs_enqueued"), res.get("pending_triage_total"))

            # 2. cheap gate — decide whether any LLM work is warranted.
            has_triage, has_sugg = _pending_kinds()

            # 3+4. fan out: ONE maintain (C1) in parallel with up to
            # WRITE_PARALLELISM writes per batch; drain writes in batches
            # until empty or DRAIN_MAX. The DB locks make this safe:
            #   FOR UPDATE SKIP LOCKED -> no double-claim on triage/suggestion
            #   try_wiki_lock(wiki_id)  -> same-wiki writer contenders skip
            # vLLM continuous-batches the concurrent inferences on the GPU.
            with ThreadPoolExecutor(max_workers=WRITE_PARALLELISM + 1) as pool:
                maintain_f = (pool.submit(_post, "/api/v1/wiki/maintain", AGENT_TIMEOUT)
                              if has_triage else None)
                done = 0
                while has_sugg and done < DRAIN_MAX:
                    batch = min(WRITE_PARALLELISM, DRAIN_MAX - done)
                    fs = [pool.submit(_post, "/api/v1/wiki/write", AGENT_TIMEOUT)
                          for _ in range(batch)]
                    any_written = False
                    for f in fs:
                        res = f.result()
                        done += 1
                        if res and res.get("written"):
                            any_written = True
                            log.info("write: wiki=%s mode=%s rev=%s",
                                     res.get("wiki_id"), res.get("mode"), res.get("revision"))
                    if not any_written:
                        break  # queue empty or all targets locked -> stop draining
                if maintain_f is not None:
                    res = maintain_f.result()
                    if res and res.get("claimed"):
                        log.info("maintain: %s", res.get("result"))

            # 5. nothing pending -> no LLM call happened this tick (free).
        except Exception as e:
            log.exception("loop error: %s", e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
