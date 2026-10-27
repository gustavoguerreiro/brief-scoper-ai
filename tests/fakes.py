"""Scripted stand-ins for the generator, used to test the whole eval pipeline offline.

OracleModel behaves like a well-behaved generator: it quotes real source text and derives the
estimate from the cited project's actual hours. FabricatingModel invents its quotes. Running both
through the real harness proves the gate can pass AND can fail, without any API call.
"""

import json
import re

from scoper.evals.dataset import EvalCase
from scoper.llm.client import ModelRequest, ModelResponse

_SOURCE = re.compile(
    r'<source id="([^"]+)"[^>]*? actual_hours="([^"]*)"[^>]*>\n(.*?)\n</source>', re.DOTALL
)
SUMMARY = "A realistic summary of what the client wants and which past project is closest."


class OracleModel:
    fabricate = False

    def __init__(self, cases: list[EvalCase]):
        self.abstain_briefs = [c.brief for c in cases if c.expect_abstain]

    def _reply(self, content: str) -> dict:
        base = {
            "summary": SUMMARY,
            "open_questions": ["Which systems must it integrate with?"],
            "confidence": "medium",
        }
        usable = []
        for cid, hours, text in _SOURCE.findall(content):
            try:
                usable.append((cid, float(hours), text))
            except ValueError:
                continue
        if any(b in content for b in self.abstain_briefs) or not usable:
            return {
                **base,
                "risks": [],
                "estimate": None,
                "confidence": "low",
                "insufficient_info": ["No comparable past project."],
            }
        cid, hours, text = usable[0]
        quote = (
            "invented words that exist nowhere" if self.fabricate else " ".join(text.split()[:6])
        )
        cite = [{"chunk_id": cid, "quote": quote}]
        return {
            **base,
            "risks": [{"description": "Integration effort", "severity": "high", "citations": cite}],
            "estimate": {
                "low": hours * 0.8,
                "high": hours * 1.2,
                "unit": "hours",
                "assumptions": ["same scope"],
                "citations": cite,
            },
            "insufficient_info": [],
        }

    def complete(self, req: ModelRequest) -> ModelResponse:
        text = json.dumps(self._reply(req.messages[0]["content"]))
        return ModelResponse(
            text=text,
            model="fake-model",
            input_tokens=500,
            output_tokens=200,
            stop_reason="end_turn",
        )


class FabricatingModel(OracleModel):
    fabricate = True
