"""The single seam for every external model call.

Layers (outermost first), composed in `build_client`:
    RetryingClient -> LoggedClient -> raw client (AnthropicClient | FakeModel)

Retry wraps the logger so every attempt, including failed ones, is recorded.
"""

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("scoper.llm")

PRICES_PATH = Path(__file__).resolve().parents[3] / "config" / "prices.json"


class TransientModelError(Exception):
    """Retryable failure (rate limit, 5xx, network)."""


class PermanentModelError(Exception):
    """Non-retryable failure (bad request, auth)."""


@dataclass(frozen=True)
class ModelRequest:
    model: str
    system: str
    messages: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    max_tokens: int = 4000
    purpose: str = "generate"
    run_id: str = ""


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float = 0.0
    request_id: str = ""
    cost_usd: float = 0.0
    stop_reason: str = ""  # end_turn | max_tokens | ... (provider value)


class ModelClient(Protocol):
    def complete(self, req: ModelRequest) -> ModelResponse: ...


def load_prices(path: Path = PRICES_PATH) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def compute_cost(prices: dict, model: str, input_tokens: int, output_tokens: int) -> float:
    p = prices.get(model)
    if p is None:
        logger.warning("no price configured for model %s; cost recorded as 0", model)
        return 0.0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


class FakeModel:
    """Deterministic offline model. Pops scripted replies in order (last one repeats)."""

    def __init__(self, replies: list[str | Exception], model: str = "fake-model"):
        self._replies = list(replies)
        self._model = model
        self.requests: list[ModelRequest] = []

    def complete(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if isinstance(reply, Exception):
            raise reply
        # crude but deterministic token estimate (~4 chars/token) so cost tests are stable
        in_chars = len(req.system) + sum(len(m["content"]) for m in req.messages)
        return ModelResponse(
            text=reply,
            model=self._model if req.model in ("", "fake-model") else req.model,
            input_tokens=max(1, in_chars // 4),
            output_tokens=max(1, len(reply) // 4),
            request_id=f"fake-{len(self.requests)}",
        )


class AnthropicClient:
    def __init__(self, api_key: str | None = None):
        import anthropic  # local import: tests and FakeModel never need the SDK

        self._anthropic = anthropic
        # max_retries=0: our RetryingClient owns retries so cost/logging see every attempt.
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=60.0)

    def complete(self, req: ModelRequest) -> ModelResponse:
        a = self._anthropic
        start = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=req.model,
                max_tokens=req.max_tokens,
                system=req.system,
                messages=req.messages,
            )
        except (a.RateLimitError, a.APIConnectionError) as e:
            raise TransientModelError(str(e)) from e
        except a.APIStatusError as e:
            if e.status_code >= 500 or e.status_code in (408, 409):
                raise TransientModelError(str(e)) from e
            raise PermanentModelError(str(e)) from e
        text = "".join(b.text for b in resp.content if b.type == "text")
        return ModelResponse(
            text=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            request_id=getattr(resp, "_request_id", "") or "",
            stop_reason=resp.stop_reason or "",
        )


@dataclass
class CallRecord:
    run_id: str
    purpose: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    status: str  # ok | error
    error: str = ""
    request_id: str = ""
    stop_reason: str = ""


@dataclass
class CallLog:
    """In-memory + optional JSONL sink. Becomes the `llm_calls` table in M2."""

    path: Path | None = None
    records: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.records.append(rec)
        logger.info(
            "llm_call run=%s purpose=%s model=%s in=%d out=%d cost=$%.6f %.0fms %s",
            rec.run_id,
            rec.purpose,
            rec.model,
            rec.input_tokens,
            rec.output_tokens,
            rec.cost_usd,
            rec.latency_ms,
            rec.status,
        )
        if rec.stop_reason == "max_tokens":
            logger.warning("llm_call run=%s hit max_tokens: output truncated", rec.run_id)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__) + "\n")

    def for_run(self, run_id: str) -> list[CallRecord]:
        return [r for r in self.records if r.run_id == run_id]

    def run_cost(self, run_id: str) -> float:
        return sum(r.cost_usd for r in self.for_run(run_id))


class LoggedClient:
    def __init__(self, inner: ModelClient, log: CallLog, prices: dict):
        self._inner, self._log, self._prices = inner, log, prices

    def complete(self, req: ModelRequest) -> ModelResponse:
        start = time.perf_counter()
        try:
            resp = self._inner.complete(req)
        except Exception as e:
            self._log.add(
                CallRecord(
                    req.run_id,
                    req.purpose,
                    req.model,
                    0,
                    0,
                    0.0,
                    (time.perf_counter() - start) * 1000,
                    "error",
                    repr(e)[:300],
                )
            )
            raise
        latency = (time.perf_counter() - start) * 1000
        cost = compute_cost(self._prices, resp.model, resp.input_tokens, resp.output_tokens)
        resp = replace(resp, latency_ms=latency, cost_usd=cost)
        self._log.add(
            CallRecord(
                req.run_id,
                req.purpose,
                resp.model,
                resp.input_tokens,
                resp.output_tokens,
                cost,
                latency,
                "ok",
                request_id=resp.request_id,
                stop_reason=resp.stop_reason,
            )
        )
        return resp


class RetryingClient:
    """Exponential backoff with jitter on TransientModelError only."""

    def __init__(
        self,
        inner: ModelClient,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._inner, self._max, self._base, self._sleep = inner, max_attempts, base_delay, sleep

    def complete(self, req: ModelRequest) -> ModelResponse:
        for attempt in range(1, self._max + 1):
            try:
                return self._inner.complete(req)
            except TransientModelError:
                if attempt == self._max:
                    raise
                self._sleep(self._base * 2 ** (attempt - 1) + random.uniform(0, 0.25))
        raise AssertionError("unreachable")


def build_client(
    raw: ModelClient,
    log: CallLog,
    prices: dict | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ModelClient:
    return RetryingClient(
        LoggedClient(raw, log, prices if prices is not None else load_prices()), sleep=sleep
    )
