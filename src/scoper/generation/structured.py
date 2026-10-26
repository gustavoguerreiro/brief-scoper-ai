"""Generic structured-output loop: call, parse, validate, repair, within a cost cap."""

import json
import logging
import re
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from scoper.llm.client import CallLog, ModelClient, ModelRequest

logger = logging.getLogger("scoper.generation")


class BudgetExceeded(Exception):
    pass


class ScopeGenerationError(Exception):
    def __init__(self, message: str, partial=None):
        super().__init__(message)
        self.partial = partial  # last schema-valid object that failed extra checks, if any


class RepairNeeded(Exception):
    def __init__(self, message: str, partial=None):
        super().__init__(message)
        self.partial = partial


def extract_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(t)


def parse_model(
    schema: type[BaseModel], check: Callable[[BaseModel], list[str]] | None = None
) -> Callable[[str], BaseModel]:
    """Build a parser: JSON -> schema -> optional semantic checks. Raises RepairNeeded."""

    def parse(text: str):
        try:
            obj = schema.model_validate(extract_json(text))
        except (json.JSONDecodeError, ValidationError) as e:
            raise RepairNeeded(str(e)[:500]) from e
        problems = check(obj) if check else []
        if problems:
            raise RepairNeeded("; ".join(problems)[:800], partial=obj)
        return obj

    return parse


def structured_call[T](
    client: ModelClient,
    log: CallLog,
    *,
    system: str,
    messages: list[dict],
    parse: Callable[[str], T],
    run_id: str,
    model: str,
    max_repairs: int = 2,
    cost_cap_usd: float = 0.05,
    max_tokens: int = 4000,
    effort: str | None = None,
) -> T:
    last_error, partial = "", None
    for attempt in range(max_repairs + 1):
        if log.run_cost(run_id) >= cost_cap_usd:
            raise BudgetExceeded(f"run {run_id} reached cap ${cost_cap_usd:.2f}")
        resp = client.complete(
            ModelRequest(
                model=model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                purpose="generate" if attempt == 0 else "repair",
                run_id=run_id,
                effort=effort,
            )
        )
        try:
            return parse(resp.text)
        except RepairNeeded as e:
            last_error, partial = str(e), e.partial
            logger.warning(
                "validation failed run=%s attempt=%d stop=%s: %s",
                run_id,
                attempt,
                resp.stop_reason,
                last_error[:200],
            )
            messages = messages + [
                {"role": "assistant", "content": resp.text},
                {
                    "role": "user",
                    "content": (
                        "Your reply was not valid. Fix it and return ONLY the corrected JSON. "
                        f"Problems: {last_error}"
                    ),
                },
            ]
    raise ScopeGenerationError(
        f"invalid output after {max_repairs} repairs: {last_error}", partial=partial
    )
