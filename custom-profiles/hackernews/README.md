# hackernews profile (keyless example)

A complete, **keyless** custom profile — the worked example for [`../SKILL.md`](../SKILL.md).
It ingests Hacker News top stories and builds wiki pages about the companies / projects /
technologies they mention.

## What it does

- `ingestor.py` polls the public Hacker News API (no key) and drops one `hn-<id>.md` per new
  top story into `data/sources/`; the existing watcher ingests + extracts it.
- `wiki_maintainer.add.md` names pages by the entity's common name and dedups variants.
- `wiki_writer.add.md` shapes each page as a tech-entity profile (Type / Homepage / Category)
  plus a dated `current-developments` chronicle.

## Run it (no key needed)

1. In the repo-root `.env`: `CUSTOM_PROFILE=hackernews`
2. `docker compose up -d`
3. Watch it flow:
   ```bash
   ls data/sources/                     # hn-*.md files dropped by the ingestor
   curl -s "http://localhost:8000/api/v1/entities?entity_type=wiki&limit=20"
   ```

Optional tuning lives in `.env` (see `.env.example`): `HN_LIMIT`, `HN_POLL_SECONDS`,
`HN_PRUNE_DAYS`. No `.env` is required — the defaults work with no key.

## Honest note

HN gives a story **title + metadata** (no article body), so individual pages are sparser
than a per-entity feed; entity pages richen as the same company/project recurs across
stories.
