import json

from scoper.evals.dataset import load_cases
from scoper.evals.perturb import (
    inject_boilerplate,
    inject_invented_claim,
    inject_number_swap,
    summarize_planted,
    wilson,
)
from scoper.evals.runner import run_planted
from scoper.llm.client import CallLog, FakeModel, build_client

SRC = [
    {
        "id": "b1:outcome:0",
        "title": "Dental booking",
        "actual_hours": 640.0,
        "actual_cost": 58000.0,
        "text": "t",
    }
]
OUT = {
    "summary": "The closest project delivered in 640 hours for $58,000 with similar features.",
    "open_questions": ["Which therapists use the calendar?"],
    "risks": [{"description": "SMS provider delays", "severity": "high", "citations": []}],
    "estimate": {
        "low": 500,
        "high": 800,
        "unit": "hours",
        "assumptions": ["Based on 640 hours actual."],
        "citations": [],
    },
    "insufficient_info": [],
    "confidence": "medium",
}


def test_number_swap_changes_one_quoted_number_and_keeps_the_rest():
    new, applied = inject_number_swap(OUT, SRC)
    assert applied and new["summary"] != OUT["summary"]
    changed = [n for n in ("640", "58,000") if n not in new["summary"]]
    assert len(changed) == 1  # exactly one of the two quoted numbers was replaced
    assert new["open_questions"] == OUT["open_questions"]
    assert "58,000" in OUT["summary"] and "640" in OUT["summary"]  # the input was not mutated


def test_number_swap_reports_when_nothing_to_swap():
    plain = dict(
        OUT,
        summary="A similar project exists.",
        estimate=dict(OUT["estimate"], assumptions=["Scaled up."]),
    )
    assert inject_number_swap(plain, SRC)[1] is False


def test_number_swap_does_not_match_inside_other_numbers():
    plain_estimate = dict(OUT["estimate"], assumptions=["Scaled up."])
    tricky = dict(OUT, summary="It took 6400 hours and 1,640 days.", estimate=plain_estimate)
    assert inject_number_swap(tricky, SRC)[1] is False


def test_invented_claim_and_boilerplate():
    assert "Dental booking" in inject_invented_claim(OUT, SRC)["summary"]
    b = inject_boilerplate(OUT)
    assert "What is the budget?" in b["open_questions"] and all(
        not r["citations"] for r in b["risks"]
    )
    assert b["summary"] == OUT["summary"]


def test_wilson_interval():
    lo, hi = wilson(18, 20)
    assert 0.69 < lo < 0.72 and 0.97 < hi < 0.99
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # NaN when n == 0


def test_run_planted_and_summary_with_a_judge_that_flags_every_defect():
    cases = {c.id: c for c in load_cases()}
    ids = ["e01", "e02", "e03", "e04"]
    rows = [{"case_id": i, "output": OUT, "sources": SRC} for i in ids]
    fail = json.dumps({"reason": "defect", "verdict": "fail"})
    log = CallLog()
    client = build_client(
        FakeModel([fail]), log, {"fake-model": {"input": 1.0, "output": 2.0}}, sleep=lambda s: None
    )
    res = run_planted(rows, cases, ids, client, log, "fake-model", workers=1)
    assert len(res) == 8  # 4 grounding + 4 actionability
    kinds = {r["defect"] for r in res}
    assert kinds == {"number_swap", "invented_claim", "boilerplate"}
    clean = {(i, c): "pass" for i in ids for c in ("grounding", "actionability")}
    s = summarize_planted(res, clean)
    assert s["boilerplate"]["caught_as_fail"] == 4 and s["clean_grounding"]["passed"] == 4


def test_include_clean_controls_are_judged_by_the_same_judge_and_win_in_the_summary():
    cases = {c.id: c for c in load_cases()}
    ids = ["e01", "e02"]
    rows = [{"case_id": i, "output": OUT, "sources": SRC} for i in ids]
    log = CallLog()
    client = build_client(
        FakeModel([json.dumps({"reason": "r", "verdict": "pass"})]),
        log,
        {"fake-model": {"input": 1.0, "output": 2.0}},
        sleep=lambda s: None,
    )
    res = run_planted(rows, cases, ids, client, log, "fake-model", workers=1, include_clean=True)
    assert sum(r["defect"] == "clean" for r in res) == 4  # 2 cases x 2 criteria
    s = summarize_planted(res, {(i, "grounding"): "fail" for i in ids})  # passed-in map is ignored
    assert s["clean_grounding"]["passed"] == 2 and "clean" not in s
