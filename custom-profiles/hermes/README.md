# hermes profile (situational monitor — "Hormuz status")

On a schedule, asks an external **Hermes Agent** (which web-searches) for the latest **Strait of
Hormuz** status and feeds each dated answer into BrainDB as a datasource. The accumulating facts form
a central **"Hormuz status"** wiki structured for a future dashboard (short / mid / long-term +
forecast metrics), while BrainDB's default rules still build wikis for the other people / orgs /
countries mentioned — a situation-aware memory. See [`../SKILL.md`](../SKILL.md) for the mechanism.

## What it does
- `ingestor.py` asks Hermes the question in `query.md` every `HERMES_POLL_SECONDS` (default 30 min),
  as a **fresh, stateless discussion** each time, and drops the dated answer (with its web sources)
  as one `hermes-hormuz-<ts>.md` into `data/sources/`; the existing watcher ingests + extracts it.
- `query.md` (committed) tells Hermes to web-search, **bring back context**, **search multiple times
  until confident**, date everything precisely, give short/mid/long-term + the most credible source,
  flag conflicts or admit uncertainty, and **include source URLs**.
- `wiki_maintainer.add.md` routes every report to the single **"Hormuz status"** page and keeps the
  other entities on the default rules.
- `wiki_writer.add.md` shapes that page: a dashboard-parseable `metrics` block + `short-term` /
  `mid-term` / `long-term` sections, leaning on the base `timeline` + `contradictions`. Timing is
  treated as critical; conflicts are resolved by credibility or explicitly marked uncertain.

## Hermes (external dependency)
Hermes runs headless via `hermes gateway` — an **OpenAI-compatible** server (default
`http://127.0.0.1:8642`, `POST /v1/chat/completions`, bearer auth). It needs a **web-search backend**
configured (Tavily / Firecrawl / Brave / SearXNG / etc.) so it can research the live status. See the
Hermes docs. **This profile does not run Hermes; it only calls it.**

## Run it (live)
1. `cp .env.example .env`; set `HERMES_ASK_URL` + `HERMES_API_KEY` (and `HERMES_MODEL` if needed).
2. Repo-root `.env`: `CUSTOM_PROFILE=hermes` (or append, e.g. `gdrive,hermes`).
3. `docker compose up -d`; watch:
   ```bash
   docker compose logs -f profile_runner    # "asking hermes (fresh discussion)" / "wrote hermes report"
   curl -s -X POST localhost:8000/api/v1/memory/context \
     -H "Content-Type: application/json" -d '{"queries":["Hormuz status"]}'
   ```

## Test WITHOUT Hermes (dry-run)
Set `HERMES_SAMPLE_FILE=custom-profiles/hermes/sample_answer.md` in `.env` (the shipped fixture); the
ingestor uses it as the "answer" instead of calling Hermes, so you can validate ingest + the wiki
shaping (metrics, short/mid/long, timeline, contradictions) before Hermes is live.

## Dormant by default
With **no `HERMES_API_KEY`** (and no sample) the ingestor logs `idle (not configured)` and **never
calls Hermes** — safe to ship inactive. It only acts once you configure it and add `hermes` to
`CUSTOM_PROFILE`.

## Notes / limits
- Answer parsing assumes OpenAI-compatible `choices[0].message.content` (adjust if your Hermes differs).
- Timing is carried as explicit dates in content + the `timeline` section — **no schema change**.
- `HERMES_DEDUP=true` skips re-ingesting an unchanged answer (avoids ~48 near-dup facts/day); set
  `false` to record every tick as a time-series point.
- Very long answers may truncate (a known Hermes issue) — `query.md` asks for a focused report.
