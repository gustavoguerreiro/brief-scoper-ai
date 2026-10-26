"""Real-provider smoke test. Excluded by default; run with: uv run pytest -m live -s"""

import os

import pytest
from dotenv import load_dotenv

from scoper.generation.scope import generate_scope
from scoper.llm.client import AnthropicClient, CallLog, build_client

load_dotenv()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]

BRIEF = (
    "We run a 3-clinic dental group and want a patient booking web app with SMS reminders, "
    "online deposits, and a dashboard for receptionists. Launch before March. Budget unknown."
)


def test_live_scope_reports_real_cost():
    log = CallLog()
    client = build_client(AnthropicClient(), log)
    out = generate_scope(
        client, log, BRIEF, "live1", os.getenv("SCOPER_MODEL", "claude-sonnet-5-5")
    )
    recs = log.for_run("live1")
    print(
        f"\nLIVE calls={len(recs)} cost=${log.run_cost('live1'):.6f} "
        f"latency_ms={sum(r.latency_ms for r in recs):.0f} "
        f"in={sum(r.input_tokens for r in recs)} out={sum(r.output_tokens for r in recs)}"
    )
    assert out.summary
