"""
Always-on ingestion watcher. Runs as a sidecar docker service.

Polls data/sources/ for new files. For each file:
  1. POST /api/v1/entities/datasources/ingest
     - 201 (new)  -> run the fact-extraction pipeline
     - 200 (dup)  -> skip extraction silently
     - 4xx/5xx    -> move to data/sources/failed/ with a sidecar .error.txt

Fact-extraction pipeline (watcher-orchestrated):
  Phase A — one /agent/query per ~600-word chunk. Each chunk agent reads
  the chunk text directly from the prompt (no get_entity), extracts
  concrete facts, saves each via save_fact, and links each back to the
  datasource via create_relation(derived_from). Returns the list of new
  fact IDs in submit_result for the watcher to parse.

  Phase B — one /agent/query with only the fact IDs + their 1-sentence
  content prefetched by the watcher. The central review agent creates
  cross-fact relations (supports/contradicts/elaborates/similar_to),
  optionally saves a holistic thought linked to the datasource, and
  optionally runs recall_memory to link existing entities.

No agent call ever loads the full document body — every request stays
well below NIM free-tier payload limits.
"""
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests

API_URL = os.getenv("BRAINDB_API_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.getenv("INGEST_POLL_INTERVAL", "7"))
WATCH_DIR = Path(os.getenv("INGEST_WATCH_DIR", "data/sources"))
INGESTED_DIR = WATCH_DIR / "ingested"
FAILED_DIR = WATCH_DIR / "failed"

ALLOWED_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".log", ".html", ".xml"}
SKIP_NAMES = {"README.md", ".gitkeep"}

CHUNK_WORDS = 600           # target chunk size (~4-5k token context per extraction call — NIM-friendly)
CHUNK_OVERLAP = 75          # words of overlap between adjacent chunks — catches facts that span a boundary

INGEST_TIMEOUT = 60
AGENT_TIMEOUT = 600          # NIM free tier is slow/flaky on gemma-4-31b; generous timeout gives retries room to succeed
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watcher] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("watcher")


def wait_for_api(timeout: int = 90) -> bool:
    """Poll /health until the API answers or timeout elapses."""
    deadline = time.time() + timeout
    url = f"{API_URL}/health"
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def move_to(path: Path, target_dir: Path, sidecar_text: str | None = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / path.name
    # Avoid overwriting on name collision
    if dest.exists():
        stem, suffix = path.stem, path.suffix
        i = 1
        while True:
            candidate = target_dir / f"{stem}.{i}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1
    shutil.move(str(path), str(dest))
    if sidecar_text:
        sidecar = dest.with_suffix(dest.suffix + ".error.txt")
        sidecar.write_text(sidecar_text, encoding="utf-8")


def call_agent(prompt: str, max_turns: int = 8) -> str | None:
    """Send a single /agent/query call. Returns the answer string, or None on failure."""
    try:
        r = requests.post(
            f"{API_URL}/api/v1/agent/query",
            json={"query": prompt, "max_turns": max_turns},
            timeout=AGENT_TIMEOUT,
        )
    except requests.RequestException as e:
        log.warning("agent request error: %s", e)
        return None
    if r.status_code != 200:
        log.warning("agent call failed %s: %s", r.status_code, r.text[:200])
        return None
    return r.json().get("answer") or ""


def split_chunks(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into word-bounded chunks with configurable overlap.

    Always splits at whitespace, never mid-word. Each chunk (after the first)
    starts `overlap` words before the previous chunk's tail, so facts that
    straddle a boundary are still visible in at least one chunk.
    """
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_words:
        overlap = 0  # nonsensical config, fall back to no overlap
    step = chunk_words - overlap
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_words]))
        if i + chunk_words >= len(words):
            break
        i += step
    return chunks


def fetch_entity(entity_id: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/api/v1/entities/{entity_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException as e:
        log.warning("fetch entity %s failed: %s", entity_id[:8], e)
    return None


def extract_facts_from_chunk(ds_id: str, title: str, idx: int, total: int, chunk_text: str) -> list[str]:
    """Ask one agent call to extract facts from a chunk, save each via save_fact,
    and link each back to the datasource via create_relation(derived_from).
    Returns the list of new fact IDs parsed from the agent's submit_result answer.
    """
    prompt = (
        f"A document was just ingested into BrainDB.\n"
        f"- datasource_id: {ds_id}\n"
        f"- title: {title}\n"
        f"- chunk: {idx}/{total}\n\n"
        f"Below is chunk {idx}/{total} of the document between <content> markers.\n"
        f"Extract the concrete, standalone FACTS from this chunk — specific claims,\n"
        f"numbers, events, named decisions. Ignore filler, opinion, and generic\n"
        f"statements. Aim for quality over quantity: typically 3-8 facts per chunk.\n\n"
        f"For EACH fact:\n"
        f'  a) Call save_fact(content="<the fact in one sentence>", certainty=0.8,\n'
        f'     source="document", keywords=[2-4 precise tags], importance=0.6,\n'
        f'     notes="Extracted from {title} chunk {idx}/{total}"). Record the\n'
        f"     returned fact id.\n"
        f"  b) Call create_relation(from_entity_id=<fact_id>, "
        f'to_entity_id="{ds_id}", relation_type="derived_from",\n'
        f'     relevance_score=0.9, description="Fact extracted from {title}").\n\n'
        f"Do NOT call get_entity. Do NOT call update_entity on the datasource.\n"
        f"Do NOT touch the datasource content — it is read-only.\n\n"
        f"When all facts in this chunk are processed, call submit_result with\n"
        f"exactly this format so the watcher can parse it:\n"
        f'  "Saved N facts from chunk {idx}/{total}: <fact_id_1>, <fact_id_2>, ..."\n\n'
        f"<content>\n{chunk_text}\n</content>"
    )
    answer = call_agent(prompt, max_turns=40)
    if not answer:
        return []
    fact_ids = UUID_RE.findall(answer)
    # Filter out the datasource id if the model happened to echo it
    return [fid for fid in fact_ids if fid != ds_id]


def central_review(ds_id: str, title: str, fact_ids: list[str]) -> None:
    """One final agent call with only the fact IDs + one-sentence contents.
    The agent creates cross-fact relations, optionally a holistic thought, and
    optionally runs recall_memory to link existing entities. No document body.
    """
    if not fact_ids:
        log.info("central review skipped for %s: no facts extracted", ds_id[:8])
        return

    lines = []
    for fid in fact_ids:
        ent = fetch_entity(fid)
        if not ent:
            continue
        content = (ent.get("content") or "").replace("\n", " ").strip()
        lines.append(f"- {fid}: \"{content[:240]}\"")
    if not lines:
        log.warning("central review aborted for %s: could not fetch any facts", ds_id[:8])
        return

    facts_block = "\n".join(lines)
    prompt = (
        f"The following facts were just extracted from document '{title}' "
        f"(datasource_id: {ds_id}):\n\n"
        f"{facts_block}\n\n"
        f"Review this set of facts holistically:\n"
        f"1. Create relations BETWEEN facts where appropriate, using create_relation\n"
        f"   with relation_type of: supports, contradicts, elaborates, similar_to,\n"
        f"   or is_example_of. Only create relations that are genuinely meaningful.\n"
        f"2. If the set as a whole suggests a broader observation or inference that\n"
        f"   none of the individual facts capture, call save_thought with that\n"
        f"   thought (certainty=0.6-0.8, source='agent-inference') and then\n"
        f'   create_relation(thought_id, "{ds_id}", "elaborates").\n'
        f"3. Optionally run recall_memory with 1-2 queries derived from the facts\n"
        f"   to find related EXISTING entities in memory. If any are clearly\n"
        f"   related, link them with tagged_with or refers_to.\n\n"
        f"Do NOT call get_entity — all facts are listed above. Do NOT touch the\n"
        f"datasource content.\n\n"
        f"When done, call submit_result with a short summary of what you added."
    )
    answer = call_agent(prompt, max_turns=30)
    if answer is None:
        log.warning("central review failed for %s", ds_id[:8])
    else:
        log.info("central review done for %s: %s", ds_id[:8], answer.replace("\n", " ")[:200])


def enrich_datasource(ds: dict, file_path: Path) -> None:
    """Orchestrate fact extraction: Phase A (per-chunk) + Phase B (central review)."""
    ds_id = ds["id"]
    title = ds.get("title") or file_path.name
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.warning("enrichment read failed for %s: %s", ds_id[:8], e)
        return

    chunks = split_chunks(text)
    if not chunks:
        log.info("enrichment skipped for %s: empty content", ds_id[:8])
        return
    log.info("extraction started for %s: %d chunks", ds_id[:8], len(chunks))

    all_fact_ids: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        log.info("extracting facts chunk %d/%d for %s", i, len(chunks), ds_id[:8])
        fact_ids = extract_facts_from_chunk(ds_id, title, i, len(chunks), chunk)
        if fact_ids:
            log.info("chunk %d/%d saved %d facts", i, len(chunks), len(fact_ids))
            all_fact_ids.extend(fact_ids)
        else:
            log.warning("chunk %d/%d produced no facts", i, len(chunks))

    log.info("extraction complete for %s: %d facts total", ds_id[:8], len(all_fact_ids))
    central_review(ds_id, title, all_fact_ids)


def process_file(path: Path) -> None:
    # Move first, then ingest. This way the file_path stored in the DB points
    # to the actual final location (data/sources/ingested/...) instead of the
    # watch folder where the file no longer lives after processing.
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    dest = INGESTED_DIR / path.name
    if dest.exists():
        stem, suffix = path.stem, path.suffix
        i = 1
        while True:
            candidate = INGESTED_DIR / f"{stem}.{i}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1
    shutil.move(str(path), str(dest))
    rel = f"data/sources/ingested/{dest.name}"

    try:
        r = requests.post(
            f"{API_URL}/api/v1/entities/datasources/ingest",
            json={
                "file_path": rel,
                "keywords": [],
                "importance": 0.6,
                "source": "document",
            },
            timeout=INGEST_TIMEOUT,
        )
    except requests.RequestException as e:
        log.warning("ingest request failed for %s: %s (moving back to retry)", dest.name, e)
        # Move back so the next tick can retry
        shutil.move(str(dest), str(WATCH_DIR / dest.name))
        return

    if r.status_code == 201:
        data = r.json()
        log.info(
            "ingested NEW: %s -> %s words=%s",
            dest.name, data["id"][:8], data.get("word_count"),
        )
        enrich_datasource(data, dest)
    elif r.status_code == 200:
        data = r.json()
        log.info("ingested DUP: %s -> %s (already existed, file kept in ingested/)", dest.name, data["id"][:8])
    else:
        log.warning("ingest failed %s: %s", r.status_code, r.text[:200])
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failed_dest = FAILED_DIR / dest.name
        shutil.move(str(dest), str(failed_dest))
        failed_dest.with_suffix(failed_dest.suffix + ".error.txt").write_text(
            f"HTTP {r.status_code}\n\n{r.text}", encoding="utf-8"
        )


def scan_once() -> None:
    for path in sorted(WATCH_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.name.startswith("."):
            continue
        if path.suffix.lower() not in ALLOWED_EXTS:
            log.info("skipping %s (unsupported extension)", path.name)
            move_to(
                path, FAILED_DIR,
                sidecar_text=(
                    f"Unsupported extension: {path.suffix}\n\n"
                    f"The watcher only ingests text files: {sorted(ALLOWED_EXTS)}.\n"
                    "PDF/docx support is not implemented yet."
                ),
            )
            continue
        process_file(path)


def main() -> None:
    if not WATCH_DIR.exists():
        log.error("watch dir not found: %s", WATCH_DIR.resolve())
        sys.exit(1)

    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("waiting for API at %s ...", API_URL)
    if not wait_for_api():
        log.error("API never came up; exiting")
        sys.exit(1)
    log.info("watcher ready (poll=%ss, dir=%s)", POLL_INTERVAL, WATCH_DIR.resolve())

    while True:
        try:
            scan_once()
        except Exception as e:
            log.exception("loop error: %s", e)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
