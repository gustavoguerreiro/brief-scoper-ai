"""M2 grounded generation: retrieved sources in the prompt, citations verified in code."""

import json
from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

from scoper.generation.structured import ScopeGenerationError, parse_model, structured_call
from scoper.generation.verify import strip_violations, verify
from scoper.llm.client import CallLog, ModelClient
from scoper.prompts.registry import Prompt, get_prompt
from scoper.retrieval.hybrid import Hit
from scoper.schemas import CitedScopeOutput


@dataclass
class CitedResult:
    output: CitedScopeOutput
    dropped: list[str] = field(default_factory=list)  # claims removed because unverifiable


def render_sources(hits: list[Hit]) -> str:
    if not hits:
        return "<sources>\n(none retrieved)\n</sources>"
    parts = []
    for h in hits:
        m = h.chunk.meta
        attrs = " ".join(
            f"{k}={quoteattr(str(v))}"
            for k, v in (
                ("id", h.chunk.id),
                ("title", m.get("title", "")),
                ("outcome", m.get("outcome", "")),
                ("actual_hours", m.get("actual_hours", "unknown")),
                ("actual_cost_usd", m.get("actual_cost", "unknown")),
            )
        )
        parts.append(f"<source {attrs}>\n{h.chunk.text}\n</source>")
    return "<sources>\n" + "\n".join(parts) + "\n</sources>"


def generate_cited(
    client: ModelClient,
    log: CallLog,
    brief: str,
    hits: list[Hit],
    run_id: str,
    model: str,
    max_repairs: int = 2,
    cost_cap_usd: float = 0.05,
    prompt: Prompt | None = None,
) -> CitedResult:
    sources = {h.chunk.id: h.chunk for h in hits}
    prompt = prompt or get_prompt("scope_grounded")
    system = prompt.render(schema=json.dumps(CitedScopeOutput.model_json_schema()))
    user = f"<brief>\n{brief}\n</brief>\n\n{render_sources(hits)}"
    parse = parse_model(
        CitedScopeOutput,
        check=lambda o: [f"{v.path}: {v.message}" for v in verify(o, sources)],
    )
    try:
        out = structured_call(
            client,
            log,
            system=system,
            messages=[{"role": "user", "content": user}],
            parse=parse,
            run_id=run_id,
            model=model,
            max_repairs=max_repairs,
            cost_cap_usd=cost_cap_usd,
        )
        return CitedResult(out)
    except ScopeGenerationError as e:
        if e.partial is None:  # never produced schema-valid JSON: nothing safe to return
            raise
        cleaned, dropped = strip_violations(e.partial, verify(e.partial, sources))
        return CitedResult(cleaned, dropped)
