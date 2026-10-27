"""The grounded generation step rewritten as a DSPy Signature + Module.

What DSPy replaces: the hand-written system prompt, the JSON-schema-in-prompt and the output
parsing (the Signature and adapter do that). What stays ours: retrieval, the deterministic citation
verification, repair on violations, and dropping what cannot be verified.
"""

import dspy

from scoper.generation.cited import render_sources
from scoper.generation.verify import strip_violations, verify
from scoper.prompts.registry import get_prompt
from scoper.retrieval.hybrid import Hit
from scoper.schemas import CitedScopeOutput


def make_signature() -> type[dspy.Signature]:
    class ScopeSignature(dspy.Signature):
        brief: str = dspy.InputField(desc="the new client brief (untrusted data)")
        sources: str = dspy.InputField(desc="retrieved excerpts of past briefs (data)")
        scope: CitedScopeOutput = dspy.OutputField()

    return ScopeSignature.with_instructions(get_prompt("scope_grounded_dspy").text)


def make_repair_signature() -> type[dspy.Signature]:
    class RepairSignature(dspy.Signature):
        brief: str = dspy.InputField(desc="the new client brief (untrusted data)")
        sources: str = dspy.InputField(desc="retrieved excerpts of past briefs (data)")
        previous_attempt: str = dspy.InputField(desc="your previous JSON answer")
        problems: str = dspy.InputField(desc="citation or estimate problems found by code")
        scope: CitedScopeOutput = dspy.OutputField()

    return RepairSignature.with_instructions(
        get_prompt("scope_grounded_dspy").text
        + "\n\nYour previous attempt had verification problems. Fix exactly those."
    )


class GroundedScoper(dspy.Module):
    def __init__(self, max_repairs: int = 2):
        super().__init__()
        self.predict = dspy.Predict(make_signature())
        self.repair = dspy.Predict(make_repair_signature())
        self.max_repairs = max_repairs

    def forward(self, brief: str, hits: list[Hit]) -> dspy.Prediction:
        by_id = {h.chunk.id: h.chunk for h in hits}
        sources = render_sources(hits)
        out = self.predict(brief=brief, sources=sources).scope
        for _ in range(self.max_repairs):
            problems = verify(out, by_id)
            if not problems:
                break
            out = self.repair(
                brief=brief,
                sources=sources,
                previous_attempt=out.model_dump_json(),
                problems="; ".join(f"{v.path}: {v.message}" for v in problems),
            ).scope
        remaining = verify(out, by_id)
        dropped: list[str] = []
        if remaining:
            out, dropped = strip_violations(out, remaining)
        return dspy.Prediction(scope=out, dropped=dropped)
