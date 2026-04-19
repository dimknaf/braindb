# data/sources

Drop files here to ingest them into BrainDB as `datasource` entities.

## Auto-ingest (always on)

The `watcher` sidecar service polls this directory every ~7 seconds. Any new file gets ingested and enriched automatically — no action required beyond dropping the file. Processed files are moved to `data/sources/ingested/`; failures go to `data/sources/failed/` with a `.error.txt` sidecar explaining why.

What enrichment means: the watcher splits the file into ~600-word chunks and, for each chunk, asks the agent to extract atomic facts. Each fact is saved as a `fact` entity and linked back to the source datasource via a `derived_from` relation. A final "central review" pass creates cross-fact relations (`supports` / `contradicts` / `elaborates` / `similar_to`) and an optional holistic observation. Watch it happen with:

```bash
docker logs braindb_watcher -f
```

## Manual ingest (if you prefer explicit control)

```bash
curl -X POST http://localhost:8000/api/v1/entities/datasources/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/sources/article.md", "keywords": ["finance"], "importance": 0.7, "source": "document"}'
```

The endpoint is idempotent by content hash — re-ingesting the same bytes returns `200` with the existing entity instead of creating a duplicate.

## Supported file types

Text files only: `.md .txt .json .yaml .yml .csv .log .html .xml`. Anything else (PDF, docx, binary) gets moved to `failed/` with an "unsupported extension" note. PDF/docx support isn't implemented yet.

## Notes

- Only `.gitkeep` and this `README.md` are tracked in git — all other files here are gitignored
- Maximum file size: 5 MB per file
- The ingest endpoint reads the file content, computes a SHA256 hash, counts words, and creates a `datasource` entity with `file_path` pointing to the original file
