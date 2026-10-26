import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from scoper.generation.scope import BudgetExceeded, ScopeGenerationError, generate_scope
from scoper.llm.client import (
    AnthropicClient,
    CallLog,
    FakeModel,
    ModelClient,
    PermanentModelError,
    TransientModelError,
    build_client,
)
from scoper.schemas import ScopeRequest, ScopeResponse

logger = logging.getLogger("scoper.api")

DEFAULT_MODEL = "claude-sonnet-5-5"
FAKE_REPLY = (
    '{"summary":"Demo output (FakeModel).","open_questions":["What is the deadline?"],'
    '"risks":[{"description":"Unclear scope","severity":"medium"}],"estimate":null,'
    '"insufficient_info":["No past projects available"],"confidence":"low"}'
)


def create_app(
    raw_client: ModelClient | None = None,
    log: CallLog | None = None,
    model: str | None = None,
    cost_cap_usd: float | None = None,
) -> FastAPI:
    app = FastAPI(title="Client Brief Scoper")
    log = log or CallLog(path=Path("logs/llm_calls.jsonl"))
    model = model or os.getenv("SCOPER_MODEL", DEFAULT_MODEL)
    cap = (
        cost_cap_usd
        if cost_cap_usd is not None
        else float(os.getenv("SCOPER_COST_CAP_USD", "0.05"))
    )
    if raw_client is None:
        raw_client = FakeModel([FAKE_REPLY]) if os.getenv("SCOPER_USE_FAKE") else AnthropicClient()
    client = build_client(raw_client, log)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "request %s %s -> %d %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    @app.post("/briefs/scope", response_model=ScopeResponse)
    def scope(req: ScopeRequest) -> ScopeResponse:
        run_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            output = generate_scope(client, log, req.brief, run_id, model, cost_cap_usd=cap)
        except BudgetExceeded as e:
            raise HTTPException(429, str(e))
        except ScopeGenerationError as e:
            raise HTTPException(502, str(e))
        except (TransientModelError, PermanentModelError) as e:
            raise HTTPException(503, f"model unavailable: {e}")
        calls = log.for_run(run_id)
        return ScopeResponse(
            run_id=run_id,
            output=output,
            cost_usd=round(log.run_cost(run_id), 6),
            llm_calls=len(calls),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    return app


def app_factory() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    return create_app()
