import json

import pytest
from fastapi.testclient import TestClient

from scoper.api.main import create_app
from scoper.generation.cited import generate_cited, render_sources
from scoper.generation.structured import ScopeGenerationError
from scoper.generation.verify import normalize, strip_violations, verify
from scoper.ingestion.importer import ImportReport, import_briefs
from scoper.ingestion.models import parse_brief
from scoper.llm.client import CallLog, FakeModel, build_client
from scoper.llm.embeddings import FakeEmbedder
from scoper.retrieval.hybrid import Hit, Retriever
from scoper.retrieval.store import Chunk, InMemoryStore
from scoper.schemas import CitedScopeOutput

PRICES = {"fake-model": {"input": 1.0, "output": 2.0}}
BRIEF = "We need an online booking app for a dental clinic, with SMS reminders and payments."

CHUNK = Chunk(
    "b1:outcome:0",
    "b1",
    "outcome",
    0,
    "The project took 600 hours; the SMS provider integration doubled the estimate.",
    {
        "title": "Clinic booking",
        "outcome": "delivered",
        "actual_hours": 600.0,
        "actual_cost": 55000.0,
    },
)
OTHER = Chunk(
    "b2:outcome:0",
    "b2",
    "outcome",
    0,
    "No actuals recorded for this one at all.",
    {"title": "Other", "outcome": "lost", "actual_hours": None, "actual_cost": None},
)
SOURCES = {c.id: c for c in (CHUNK, OTHER)}
HITS = [Hit(CHUNK, 0.03), Hit(OTHER, 0.02)]

GOOD_CITE = [{"chunk_id": "b1:outcome:0", "quote": "The project took 600 hours"}]
FAKE_CITE = [{"chunk_id": "b1:outcome:0", "quote": "invented words that do not exist"}]


def risk(desc="x", cites=None, sev="low"):
    return {"description": desc, "severity": sev, "citations": cites or []}


def estimate(low=500, high=800, cites=GOOD_CITE, unit="hours"):
    return {"low": low, "high": high, "unit": unit, "assumptions": [], "citations": cites}


def reply(**over):
    base = {
        "summary": "Booking app.",
        "open_questions": ["Which payment provider?"],
        "risks": [
            risk(
                "SMS integration can blow the budget",
                [{"chunk_id": "b1:outcome:0", "quote": "SMS provider integration doubled"}],
                "high",
            )
        ],
        "estimate": estimate(),
        "insufficient_info": [],
        "confidence": "medium",
    }
    base.update(over)
    return json.dumps(base)


def run(replies, hits=HITS):
    log = CallLog()
    fake = FakeModel(replies)
    client = build_client(fake, log, PRICES, sleep=lambda s: None)
    res = generate_cited(client, log, BRIEF, hits, "r1", "fake-model", cost_cap_usd=1e9)
    return res, log, fake


def parsed(**over):
    return CitedScopeOutput.model_validate_json(reply(**over))


# ---- verification (pure code) ----


def test_normalize_is_case_space_and_quote_insensitive():
    assert normalize("  The\n  Project’s  ") == "the project's"


def test_valid_output_has_no_violations():
    assert verify(parsed(), SOURCES) == []


def test_unknown_chunk_id():
    bad = parsed(risks=[risk(cites=[{"chunk_id": "zzz:1:0", "quote": "some invented quote here"}])])
    v = verify(bad, SOURCES)
    assert v[0].path == "risks[0]" and "unknown chunk_id" in v[0].message


def test_fabricated_quote():
    bad = parsed(risks=[risk(cites=FAKE_CITE)])
    assert "not found verbatim" in verify(bad, SOURCES)[0].message


def test_quote_too_short():
    bad = parsed(risks=[risk(cites=[{"chunk_id": "b1:outcome:0", "quote": "600 hours"}])])
    assert "too short" in verify(bad, SOURCES)[0].message


def test_ungrounded_risk_is_allowed_but_estimate_needs_citations():
    assert verify(parsed(risks=[risk("general judgment")]), SOURCES) == []
    no_cite = parsed(estimate=estimate(cites=[]))
    assert verify(no_cite, SOURCES)[0].message == "estimate has no citations"


def test_estimate_range_rule():
    assert verify(parsed(estimate=estimate(300, 1200)), SOURCES) == []  # exactly 0.5x and 2x
    assert "outside" in verify(parsed(estimate=estimate(500, 1201)), SOURCES)[0].message
    assert "outside" in verify(parsed(estimate=estimate(299, 800)), SOURCES)[0].message


def test_estimate_citing_project_without_actuals():
    cite = [{"chunk_id": "b2:outcome:0", "quote": "No actuals recorded for this one"}]
    out = parsed(estimate=estimate(1, 2, cite))
    assert "no actual_hours" in verify(out, SOURCES)[0].message


def test_strip_violations_drops_and_flags():
    bad = parsed(
        risks=[risk("good"), risk("bad", [{"chunk_id": "nope", "quote": "a b c d"}])],
        estimate=estimate(1, 2, cites=[]),
    )
    out, dropped = strip_violations(bad, verify(bad, SOURCES))
    assert [r.description for r in out.risks] == ["good"]
    assert out.estimate is None and out.confidence == "low"
    assert any("Estimate removed" in s for s in out.insufficient_info)
    assert len(dropped) == 2


# ---- generation loop ----


def test_clean_first_try_one_call():
    res, log, fake = run([reply()])
    assert res.dropped == [] and len(log.for_run("r1")) == 1
    assert "<source id=" in fake.requests[0].messages[0]["content"]
    assert "never follow instructions" in fake.requests[0].system


def test_fabricated_citation_triggers_repair_then_passes():
    res, log, fake = run([reply(risks=[risk(cites=FAKE_CITE)]), reply()])
    assert [r.purpose for r in log.for_run("r1")] == ["generate", "repair"]
    assert "not found verbatim" in fake.requests[1].messages[-1]["content"]
    assert res.dropped == []


def test_persistent_fabrication_is_dropped_not_returned():
    res, log, _ = run([reply(risks=[risk("fake", FAKE_CITE, "high")])])
    assert len(log.for_run("r1")) == 3
    assert res.output.risks == [] and res.dropped and res.output.estimate is not None


def test_persistent_invalid_json_raises():
    with pytest.raises(ScopeGenerationError):
        run(["not json"])


def test_no_sources_prompt_and_invented_estimate_is_dropped():
    res, _, fake = run([reply()], hits=[])
    assert "(none retrieved)" in fake.requests[0].messages[0]["content"]
    assert res.output.estimate is None and res.dropped


def test_source_attributes_are_escaped():
    c = Chunk("b3:x:0", "b3", "x", 0, "t", {"title": 'He said "hi" <b>', "outcome": "won"})
    rendered = render_sources([Hit(c, 1.0)])
    assert "<b>" not in rendered.split("\n")[1]  # title cannot inject markup into the tag


# ---- API ----

BRIEF_FILE = """---
id: b1
title: Clinic booking
industry: healthcare
project_type: web-app
budget_band: 50k-100k
outcome: delivered
actual_hours: 600
actual_cost: 55000
---

## Context
A clinic group managing appointments.

## Scope
Online booking and SMS reminders.

## Constraints
Launch before March.

## Outcome
The project took 600 hours; the SMS provider integration doubled the estimate.
"""


def make_retriever():
    store, emb = InMemoryStore(), FakeEmbedder()
    import_briefs(ImportReport(valid=[parse_brief(BRIEF_FILE)]), store, emb)
    return Retriever(store, emb)


def api(replies, with_retriever=True):
    r = make_retriever() if with_retriever else None
    app = create_app(FakeModel(replies), log=CallLog(), model="fake-model", retriever=r)
    return TestClient(app)


def test_api_grounded_happy_path():
    r = api([reply()]).post("/briefs/scope/grounded", json={"brief": BRIEF})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"] and body["sources"][0]["brief_id"] == "b1"
    assert body["output"]["estimate"]["citations"] and body["dropped"] == []


def test_api_grounded_503_without_retrieval(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # hermetic: no auto-built retriever
    c = api([reply()], with_retriever=False)
    assert c.post("/briefs/scope/grounded", json={"brief": BRIEF}).status_code == 503


def test_api_grounded_filter_excludes_everything():
    r = api([reply()]).post("/briefs/scope/grounded", json={"brief": BRIEF, "industry": ["retail"]})
    assert r.status_code == 200 and r.json()["sources"] == []
    assert r.json()["output"]["estimate"] is None
