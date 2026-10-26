# Client Brief Scoper

Paste a client brief, get a structured scope (summary, open questions, risks, estimate range), reviewed by a human. See `docs/DESIGN.md` for the design, `RESULTS.md` for measured numbers, `docs/STUDY.md` for interview notes.

## Run (M1)
```bash
uv sync
uv run pytest                      # offline, FakeModel only
SCOPER_USE_FAKE=1 uv run uvicorn scoper.api.main:app_factory --factory   # offline demo server
uv run pytest -m live -s           # real model; needs ANTHROPIC_API_KEY in .env
```
`POST /briefs/scope {"brief": "..."}`. Every model call goes through `src/scoper/llm/client.py`; every call is logged to `logs/llm_calls.jsonl` with tokens, cost and latency.

## M2: grounded scoping
```bash
docker compose up -d --wait db                                    # Postgres + pgvector on localhost:5440
cp data/briefs/_TEMPLATE.md data/briefs/b001.md                   # write your briefs from the template
uv run python -m scoper.ingestion.importer data/briefs --dry-run  # validate (no DB, no key)
uv run python -m scoper.ingestion.importer data/briefs            # embed + store (needs OPENAI_API_KEY, DATABASE_URL)
uv run uvicorn scoper.api.main:app_factory --factory              # POST /briefs/scope/grounded
uv run pytest -m db                                               # tests against the Compose Postgres
```
