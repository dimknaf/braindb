<!-- hackernews profile: tech-entity chronicle structure (appended). -->

## Tech-entity format (companies / projects / technologies / people)

These pages are built from Hacker News stories. Keep EVERY base rule above —
`[[ref:UUID]]` citations, the `<!-- section:NAME -->` markers, and the
`<!-- wiki:meta ... -->` header — and shape the page like this so every entity reads
consistently:

- **Title** = the entity's common name, e.g. `# OpenAI`.
- **Meta keywords** = the entity name first, then key themes.

Use these sections in addition to the base `contradictions` / `references`:

```
<!-- section:profile -->
**Type:** company / project / technology / person
**Homepage/Repo:** <url> [[ref:UUID]]
**Category:** <area, e.g. AI, databases, web> [[ref:UUID]]
<!-- section:background -->          durable facts about the entity
<!-- section:current-developments --> dated HN items, NEWEST FIRST, each with [[ref:UUID]]
```

**Profile rules** — one labeled field per line. Fill **Homepage/Repo** and **Category**
ONLY if a source supports it (and append its `[[ref:UUID]]`); otherwise write `(unknown)`
— **never invent**. **Type** is your best source-supported classification.

### Chronicle aging

`current-developments` is a rolling window of what's happening NOW. As items get old (relative to
the current date given above) or superseded, demote their lasting substance into `background`
(carrying the `[[ref:UUID]]` along) and drop the stale line from `current-developments`. Never
silently drop a cited fact — move its citation to `background`. Dated items use one consistent
format: `- 2026-06-20 — <what happened> [[ref:UUID]]` (ISO date, em-dash, claim, citation).
