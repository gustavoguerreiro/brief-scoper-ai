import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from scoper.generation.cited import generate_cited
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
from scoper.retrieval.hybrid import Retriever
from scoper.retrieval.store import Filters
from scoper.schemas import (
    GroundedRequest,
    GroundedResponse,
    ScopeRequest,
    ScopeResponse,
    SourceInfo,
)

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
    retriever: Retriever | None = None,
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
    if retriever is None and os.getenv("OPENAI_API_KEY") and not os.getenv("SCOPER_USE_FAKE"):
        from scoper.llm.embeddings import OpenAIEmbedder
        from scoper.retrieval.pg_store import PgStore, database_url

        embedder = OpenAIEmbedder(log)
        retriever = Retriever(PgStore(database_url(), embedder.dim), embedder)

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

    @app.post("/briefs/scope/grounded", response_model=GroundedResponse)
    def scope_grounded(req: GroundedRequest) -> GroundedResponse:
        if retriever is None:
            raise HTTPException(
                503, "retrieval is not configured (set OPENAI_API_KEY and start the database)"
            )
        run_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        filters = Filters(industry=tuple(req.industry), project_type=tuple(req.project_type))
        try:
            hits = retriever.search(req.brief, filters, k=req.top_k, run_id=run_id)
            result = generate_cited(client, log, req.brief, hits, run_id, model, cost_cap_usd=cap)
        except BudgetExceeded as e:
            raise HTTPException(429, str(e))
        except ScopeGenerationError as e:
            raise HTTPException(502, str(e))
        except (TransientModelError, PermanentModelError) as e:
            raise HTTPException(503, f"model unavailable: {e}")
        return GroundedResponse(
            run_id=run_id,
            output=result.output,
            sources=[
                SourceInfo(
                    chunk_id=h.chunk.id,
                    brief_id=h.chunk.brief_id,
                    title=h.chunk.meta.get("title", ""),
                    section=h.chunk.section,
                    score=round(h.score, 5),
                )
                for h in hits
            ],
            dropped=result.dropped,
            cost_usd=round(log.run_cost(run_id), 6),
            llm_calls=len(log.for_run(run_id)),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    return app


def app_factory() -> FastAPI:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    return create_app()
