"""M1 baseline generation: one structured call + validation-repair loop.

No retrieval, so the estimate is ungrounded; the prompt tells the model to say so.
Kept as the baseline that the grounded pipeline (cited.py) is compared against in M3.
"""

import json

from scoper.generation.structured import (  # noqa: F401  (re-exported for callers/tests)
    BudgetExceeded,
    ScopeGenerationError,
    parse_model,
    structured_call,
)
from scoper.llm.client import CallLog, ModelClient
from scoper.prompts.registry import Prompt, get_prompt
from scoper.schemas import ScopeOutput


def generate_scope(
    client: ModelClient,
    log: CallLog,
    brief: str,
    run_id: str,
    model: str,
    max_repairs: int = 2,
    cost_cap_usd: float = 0.05,
    prompt: Prompt | None = None,
) -> ScopeOutput:
    prompt = prompt or get_prompt("scope_baseline")
    system = prompt.render(schema=json.dumps(ScopeOutput.model_json_schema()))
    return structured_call(
        client,
        log,
        system=system,
        messages=[{"role": "user", "content": f"<brief>\n{brief}\n</brief>"}],
        parse=parse_model(ScopeOutput),
        run_id=run_id,
        model=model,
        max_repairs=max_repairs,
        cost_cap_usd=cost_cap_usd,
    )
