"""Deterministic citation checks. Code decides what is grounded, never the model.

Limit: this proves a quote EXISTS in a retrieved chunk, not that it SUPPORTS the claim.
Support is a judgment call, left to the LLM judge in M3.
"""

import re
from dataclasses import dataclass

from scoper.retrieval.store import Chunk
from scoper.schemas import Citation, CitedScopeOutput

MIN_QUOTE_WORDS = 3
# Arbitrary starting values (not tuned): an estimate may sit between half of the smallest
# and double the largest actual of the projects it cites. Revisit with the M3 eval set.
ESTIMATE_LOW_FACTOR = 0.5
ESTIMATE_HIGH_FACTOR = 2.0

_ACTUAL_FIELD = {"hours": "actual_hours", "usd": "actual_cost"}


@dataclass(frozen=True)
class Violation:
    path: str  # "risks[2]" or "estimate"
    message: str


def normalize(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def check_citation(c: Citation, sources: dict[str, Chunk]) -> str | None:
    chunk = sources.get(c.chunk_id)
    if chunk is None:
        return f"unknown chunk_id {c.chunk_id!r} (not among the provided sources)"
    quote = normalize(c.quote)
    if len(quote.split()) < MIN_QUOTE_WORDS:
        return f"quote too short (<{MIN_QUOTE_WORDS} words) for {c.chunk_id}"
    if quote not in normalize(chunk.text):
        return f"quote not found verbatim in {c.chunk_id}"
    return None


def verify(output: CitedScopeOutput, sources: dict[str, Chunk]) -> list[Violation]:
    found: list[Violation] = []
    for i, risk in enumerate(output.risks):
        for c in risk.citations:
            if msg := check_citation(c, sources):
                found.append(Violation(f"risks[{i}]", msg))
    est = output.estimate
    if est is not None:
        if not est.citations:
            found.append(Violation("estimate", "estimate has no citations"))
        bad = [m for c in est.citations if (m := check_citation(c, sources))]
        found.extend(Violation("estimate", m) for m in bad)
        if est.citations and not bad:
            found.extend(_check_range(est, sources))
    return found


def _check_range(est, sources: dict[str, Chunk]) -> list[Violation]:
    field = _ACTUAL_FIELD[est.unit]
    by_brief = {
        sources[c.chunk_id].brief_id: sources[c.chunk_id].meta.get(field) for c in est.citations
    }
    actuals = [v for v in by_brief.values() if v is not None]
    if not actuals:
        return [
            Violation(
                "estimate", f"cited projects have no {field}; cannot ground a {est.unit} estimate"
            )
        ]
    lo, hi = ESTIMATE_LOW_FACTOR * min(actuals), ESTIMATE_HIGH_FACTOR * max(actuals)
    if est.low < lo or est.high > hi:
        return [
            Violation(
                "estimate",
                f"range {est.low:g}-{est.high:g} {est.unit} is outside {lo:g}-{hi:g}, "
                f"derived from cited {field} values {sorted(actuals)}",
            )
        ]
    return []


def strip_violations(
    output: CitedScopeOutput, violations: list[Violation]
) -> tuple[CitedScopeOutput, list[str]]:
    """Drop what could not be verified and say so, instead of failing the whole request."""
    bad = {v.path for v in violations}
    dropped = [f"{v.path}: {v.message}" for v in violations]
    kept = [r for i, r in enumerate(output.risks) if f"risks[{i}]" not in bad]
    out = output.model_copy(update={"risks": kept})
    if "estimate" in bad:
        out = out.model_copy(
            update={
                "estimate": None,
                "confidence": "low",
                "insufficient_info": [
                    *out.insufficient_info,
                    "Estimate removed: it could not be verified against the cited sources.",
                ],
            }
        )
    return out, dropped
