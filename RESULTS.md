# RESULTS

Only measured values belong here. Bad results are recorded too.

| Date | Milestone | Dataset size | Metric | Value | Notes |
|---|---|---|---|---|---|
| 2026-10-26 | M1 | 16 offline tests (FakeModel) | tests passed | 16 / 16 | `uv run pytest`. Verifies logic only (retries, repair loop, cost math, budget cap, API codes). Says nothing about real model quality. |
| 2026-10-26 | M1 | 0 real calls | real cost/latency per request | **NOT MEASURED** | No `ANTHROPIC_API_KEY` in this environment. Run `uv run pytest -m live -s` after creating `.env`, then paste the printed `LIVE ...` line here. |

## Unverified inputs (not measurements)
- `config/prices.json` holds list prices copied from the claude-api skill's cache dated 2026-10-06. Confirm against the provider pricing page.
