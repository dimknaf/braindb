<!-- hermes profile: situational tracker — clean, generic dashboard layout (appended). -->

## Ongoing-situation format

When the subject is an ongoing real-world SITUATION tracked over time from dated reports (a strait's
status, a conflict, an outbreak, a market regime — anything that evolves), keep EVERY base rule —
`[[ref:UUID]]` citations, the `<!-- section:NAME -->` markers, the `<!-- wiki:meta ... -->` header — and
lay the page out in clean, consistent, separately-parseable sections. For a situational page, use
**exactly the sections below, in this order** — these **REPLACE** the base "Recommended structure" list
(no `overview` / `timeline` / `sources`); add the base `contradictions` only when sources disagree; do
not invent extra sections.

All windows below are RETROSPECTIVE — they look BACK from **today** (the current date given above). Only
`forecast` looks forward, and it is ONE outlook (never split into short/long-term tiers). Bucket each
event by its OWN date relative to today.

```
<!-- section:status -->
**Current status:** <short phrase, in the situation's own terms>   [[ref:UUID]]
**As of:** <date the CURRENT status was actually observed, from the report content>   [[ref:UUID]]
**Last checked:** <the latest report's fetch date>
**Confidence:** high | medium | low — <one-line basis>
**Most credible source:** <name> — <url>   [[ref:UUID]]
**Open conflict:** none | <one-line description>

<!-- section:now -->          today / latest perception — current state + most recent reporting, dated
<!-- section:recent-days -->  the last ~3-4 days before today — what changed, newest first, dated
<!-- section:recent-weeks --> the last ~1-2 weeks before today — broader trajectory, dated
<!-- section:forecast -->     the ONLY forward-looking section — ONE short outlook with its time
                             horizon + confidence (a single statement; do NOT split into multiple
                             short/long-term forecasts)
<!-- section:history -->      running dated log of key events, NEWEST FIRST; grows with every report
```

`status` is **one labelled field per line** — a dashboard parses it. Give a field a value **and** its
`[[ref:UUID]]` only when a source supports it; otherwise write `(unknown)` — **never invent**.

**Dated lines** use ONE consistent format, newest first: `- 2026-06-20 — <what happened> [[ref:UUID]]`
(ISO date, em-dash, claim, citation).

**Rules (generic — adapt to any situation, do not over-fit):**
- Use the situation's **own vocabulary** for status; don't force fixed categories.
- **Date every claim** ("as of <date>"); `Current status` = the most recent credible report.
- Show **confidence cleanly** — once in `status`, and on the single `forecast` line.
- **Conflicts:** adopt the more credible (say why, cite **both** in `contradictions`) **or** set
  `Confidence: low` and state plainly that certainty is not available — never fabricate one answer.
  Hedge unverified claims ("reportedly", "per [[ref:UUID]]").
- **Degrade gracefully:** if a period has little info, one dated line like "(no significant change
  reported)" is fine — don't pad with invented detail.

### Re-bucket every rewrite (this is YOUR job, each time)
Place every event by its OWN date relative to **today**: today/just now → `now`; within ~3-4 days →
`recent-days`; within ~1-2 weeks → `recent-weeks`; older → `history`. `history` is the durable record —
append new events (newest first), NEVER delete a cited one (keep its `[[ref:UUID]]`). So as days pass an
event moves now → recent-days → recent-weeks → history.
