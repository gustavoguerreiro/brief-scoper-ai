"""DSPy language model backed by OUR ModelClient.

DSPy would normally call providers through LiteLLM, bypassing the single client wrapper (cost
logging, retries, FakeModel in tests). This adapter keeps every call going through ModelClient, so
optimizer runs show up in the same call log and respect a hard spend cap.
"""

import contextlib
import contextvars
from collections.abc import Iterator
from types import SimpleNamespace

import dspy

from scoper.generation.structured import BudgetExceeded
from scoper.llm.client import CallLog, ModelClient, ModelRequest

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("dspy_run_id", default="dspy")


@contextlib.contextmanager
def run_scope(run_id: str) -> Iterator[None]:
    """Tag every model call made inside the block with this run id (cost per example)."""
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


class ClientLM(dspy.BaseLM):
    def __init__(
        self,
        client: ModelClient,
        log: CallLog,
        model: str,
        max_tokens: int = 4000,
        budget_usd: float | None = None,
        effort: str | None = None,
    ):
        super().__init__(model=model, model_type="chat", cache=False, num_retries=0)
        self._client, self._log = client, log
        self._max_tokens, self._budget, self._effort = max_tokens, budget_usd, effort
        self._start_spend = sum(r.cost_usd for r in log.records)

    def spent(self) -> float:
        return sum(r.cost_usd for r in list(self._log.records)) - self._start_spend

    def forward(self, prompt=None, messages=None, **kwargs):
        if self._budget is not None and self.spent() >= self._budget:
            raise BudgetExceeded(f"DSPy run reached its ${self._budget:.2f} budget")
        msgs = messages or [{"role": "user", "content": prompt or ""}]
        system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
        chat = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] != "system"]
        resp = self._client.complete(
            ModelRequest(
                model=self.model,
                system=system,
                messages=chat,
                max_tokens=self._max_tokens,
                purpose="dspy",
                run_id=_run_id.get(),
                effort=self._effort,
            )
        )
        message = SimpleNamespace(content=resp.text, tool_calls=None, reasoning_content=None)
        usage = {
            "prompt_tokens": resp.input_tokens,
            "completion_tokens": resp.output_tokens,
            "total_tokens": resp.input_tokens + resp.output_tokens,
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=resp.stop_reason or "stop")],
            usage=usage,
            model=resp.model,
        )
