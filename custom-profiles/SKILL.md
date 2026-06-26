---
name: braindb-custom-profile
description: How to author a BrainDB custom profile — prompt add/replace fragments and an optional keyless ingestor — that shapes wiki naming/structure and feeds a custom ingestion source, with zero effect on defaults when inactive.
allowed-tools: Read Write Edit Bash
---

# Authoring a BrainDB custom profile

A **custom profile** lets you (a) shape how the wiki maintainer/writer **name and structure**
pages and (b) feed BrainDB a **custom ingestion source** — with **zero effect on default
behaviour when the profile is not active**. One env switch turns a profile on; one
self-contained folder holds everything it needs.

The committed `custom-profiles/hackernews/` folder is a complete, **keyless** worked example
to copy from. `custom-profiles/README.md` is the short contract; this is the deep how-to.

## The model

- **One switch:** `CUSTOM_PROFILE` in the repo-root `.env` names the active profile(s),
  comma-separated. **Unset ⇒ every prompt is byte-identical to the baked-in default** and the
  ingestor supervisor sleeps.
- **One folder:** `custom-profiles/<name>/` holds the profile's prompt fragments, an optional
  `ingestor.py`, and an optional `.env`.

## Folder contract

```
custom-profiles/<name>/
├── wiki_maintainer.add.md      # appended to the maintainer prompt
├── wiki_maintainer.replace.md  # OR replaces it entirely (advanced)
├── wiki_writer.add.md          # appended to the writer prompt
├── wiki_writer.replace.md      # OR replaces it entirely (advanced)
├── ingestor.py                 # optional: a standalone source feeder
├── .env                        # optional: the ingestor's own config/secrets
└── README.md
```

Targets are `wiki_maintainer` and `wiki_writer`. `<target>.add.md` is **appended** to the
base prompt; `<target>.replace.md` **replaces** it. With several active profiles, `add`
fragments are concatenated in `CUSTOM_PROFILE` order. All files are optional.

### What each prompt shapes
- **`wiki_maintainer`** decides, per orphan entity: skip / create (with a `proposed_name`) /
  attach / consolidate. Shape **naming and dedup** here (e.g. "name companies by official
  name; same ticker = same company").
- **`wiki_writer`** authors the page body (free markdown). Shape **structure** here — the
  sections, a labelled profile block, a dated chronicle.

## Rules that keep it safe

1. **Prefer `add` over `replace`.** `add` keeps the base prompt — and all the machine
   contract it carries — intact and only layers your guidance on top. `replace` makes you
   re-own that contract.
2. **Never break the machine contract** (the writer body is parsed for these):
   - `[[ref:UUID]]` inline citations — how a wiki links to its evidence;
   - `<!-- section:NAME -->` markers — the section system (arbitrary NAMEs are allowed, so
     you may add e.g. `profile`, `background`, `current-developments`);
   - the `<!-- wiki:meta canonical_name=… keywords=a;b -->` header — the only source of a
     page's keywords.
3. **Cite or `(unknown)` — never invent.** A structured field gets a value **and** its
   `[[ref:UUID]]` only when a source supports it; otherwise write `(unknown)`.
4. **Make the clustering entity salient in the dropped file.** Facts cluster into a wiki by
   the keyword they're tagged with, so put the entity (ticker, company, project) prominently
   in the file (e.g. a `**Tickers:** MSFT` header line) so the extractor tags facts with it.

## The ingestor (optional)

A profile may ship `ingestor.py`: a standalone, long-running loop that feeds BrainDB. The
committed `profile_runner` sidecar launches each active profile's `ingestor.py` as an
**isolated subprocess** (restart-on-exit), and sleeps when no profile is active — so a broken
ingestor can never affect the api/watcher/wiki.

The simplest, least-coupled ingestor just **writes files into `data/sources/`** and lets the
existing watcher do ingestion + fact-extraction — no DB or agent calls. Pattern:

- load config from a sibling `.env` (a tiny parser into `os.environ`, or `python-dotenv`);
- poll your source; for each **new** item (track seen ids in a `.state/` file), write one
  `<prefix>-<id>.md` into `data/sources/`, with the entity made salient;
- prune old files under `data/sources/ingested/` (the verbatim text is stored in the DB at
  ingest, so the file is only a carrier);
- sleep and repeat.

Resolve `data/sources/` from the script's own location so it works in the container and
standalone: `Path(__file__).resolve().parents[1] / "data" / "sources"`. Load secrets from the
sibling `.env` so **no profile-specific variable ever appears in the public
`docker-compose.yml`**. Keep files small (title + a few fields).

## Activate and verify

1. (if the profile has secrets) `cp custom-profiles/<name>/.env.example custom-profiles/<name>/.env` and fill it.
2. In the repo-root `.env`: `CUSTOM_PROFILE=<name>` (or `a,b` for several).
3. `docker compose up -d` — the api picks up the prompt shaping; `profile_runner` launches the
   ingestor.
4. Watch it flow:
   ```bash
   ls data/sources/                                   # files dropped by the ingestor
   curl -s "http://localhost:8000/api/v1/entities?entity_type=wiki&limit=20"
   curl -s -X POST http://localhost:8000/api/v1/memory/context \
     -H "Content-Type: application/json" -d '{"queries":["<an entity name>"]}'
   ```

To deactivate: remove the `CUSTOM_PROFILE` line and `docker compose up -d` — defaults restored.

## Worked example

`custom-profiles/hackernews/` is a complete, **keyless** profile (Hacker News → tech-entity
wikis). Copy its `ingestor.py` + `wiki_*.add.md` as a starting point, then run it with just
`CUSTOM_PROFILE=hackernews` — no key required.

`custom-profiles/gdrive/` is a second example: a **folder follower** that ingests a Google Drive
folder **incrementally** — only new files and, on edits, only the changed sections (a `.state`
manifest + per-doc snapshots, each delta self-describing *what part it is* and *where it belongs*).
Copy it when your source is a watched folder of documents that change over time rather than a feed.

A note on privacy: keep real profiles that carry a domain prompt or an API key **gitignored**
(see `.gitignore`); only generic teaching profiles like `example/` and `hackernews/` are
committed.
