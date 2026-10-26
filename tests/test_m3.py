import json
import math

import pytest

from scoper.evals.checks import check_output, summarize_checks
from scoper.evals.dataset import EvalCase, load_cases
from scoper.evals.judge import (
    judge_pair_both_orders,
    judge_pointwise,
    position_bias_report,
    render_sources_text,
)
from scoper.evals.metrics import (
    bootstrap_ci,
    brief_ranking,
    cohen_kappa,
    confusion,
    hit_at_k,
    percent_agreement,
    reciprocal_rank,
)
from scoper.evals.retrieval_eval import CachingEmbedder, evaluate_retrieval, format_report
from scoper.evals.runner import (
    _fake_retriever,
    build_label_sample,
    calibration_report,
    pairwise_summary,
    run_generation,
    run_pairwise,
    run_pointwise,
)
from scoper.ingestion.importer import load_dir
from scoper.llm.client import CallLog, FakeModel, build_client
from scoper.llm.embeddings import FakeEmbedder
from scoper.retrieval.hybrid import Hit
from scoper.retrieval.store import Chunk

PRICES = {"fake-model": {"input": 1.0, "output": 2.0}}


def make_client(replies):
    log = CallLog()
    fake = FakeModel(replies)
    return build_client(fake, log, PRICES, sleep=lambda s: None), log, fake


# ---------------- dataset ----------------


def test_eval_set_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 20
    corpus = {d.id: d for d in load_dir(__import__("pathlib").Path("data/briefs")).valid}
    for c in cases:
        for g in c.gold_brief_ids:  # gold must exist and be visible to the default filter
            assert g in corpus and corpus[g].visibility == "team"
        if c.expect_abstain:
            assert c.gold_brief_ids == []
    tags = {t for c in cases for t in c.tags}
    assert {"typical", "near-duplicate", "injection", "vague", "out-of-domain"} <= tags
    assert {c.split for c in cases} == {"dev", "test"}
    assert all(c.canary in c.brief for c in cases if c.canary)


def test_eval_briefs_are_not_corpus_briefs():
    corpus_text = " ".join(
        d.raw_text for d in load_dir(__import__("pathlib").Path("data/briefs")).valid
    )
    assert all(c.brief not in corpus_text for c in load_cases())


# ---------------- metrics ----------------


def test_hit_and_rr():
    assert hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert hit_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert reciprocal_rank(["a", "b", "c"], ["b", "c"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_brief_ranking_keeps_best_chunk_order():
    mk = lambda cid, bid: Hit(Chunk(cid, bid, "s", 0, "t"), 1.0)
    assert brief_ranking([mk("a:1", "a"), mk("b:1", "b"), mk("a:2", "a")]) == ["a", "b"]


def test_kappa_known_values():
    assert cohen_kappa(list("ppff"), list("pfff")) == pytest.approx(0.5)
    assert cohen_kappa(list("ppff"), list("ppff")) == pytest.approx(1.0)
    assert cohen_kappa(list("pfpf"), list("fpfp")) == pytest.approx(-1.0)
    assert math.isnan(cohen_kappa(list("pppp"), list("pppp")))  # no variation: undefined
    assert percent_agreement(list("ppff"), list("pfff")) == 0.75
    assert confusion(["p", "p"], ["p", "f"]) == {"human=p|judge=f": 1, "human=p|judge=p": 1}


def test_bootstrap_ci_brackets_and_is_wide_for_small_n():
    a = list("ppppppppffffffffpppp"[:20])
    b = list("pppppppfffffffffpppf"[:20])
    k = cohen_kappa(a, b)
    lo, hi = bootstrap_ci(a, b)
    assert lo <= k <= hi and hi - lo > 0.2


# ---------------- retrieval eval (fake embedder: tests plumbing, not semantics) ----------------


def test_retrieval_eval_runs_on_real_corpus_offline():
    rep = evaluate_retrieval(_fake_retriever(), load_cases())
    assert rep["n_cases"] == 19  # only cases with gold briefs are scored
    for m in rep["configs"].values():
        assert 0 <= m["mrr"] <= 1 and m["hit@1"] <= m["hit@3"] <= m["hit@5"] <= 1
    cap = rep["configs"]["hybrid_cap2"]["distinct_briefs_in_top5_chunks"]
    assert cap >= rep["configs"]["hybrid"]["distinct_briefs_in_top5_chunks"]
    assert "hybrid_cap2" in format_report(rep)


def test_caching_embedder_embeds_each_query_once():
    inner = FakeEmbedder()
    calls = []
    orig = inner.embed
    inner.embed = lambda texts, run_id="": calls.append(list(texts)) or orig(texts, run_id)
    c = CachingEmbedder(inner)
    c.embed(["x", "y"])
    c.embed(["x"])
    assert calls == [["x", "y"]]


# ---------------- checks ----------------


def case(**kw):
    base = {
        "id": "e",
        "brief": "b",
        "tags": [],
        "gold_brief_ids": [],
        "expect_abstain": False,
        "canary": None,
        "split": "dev",
    }
    return EvalCase(**{**base, **kw})


OUT = {
    "summary": "s",
    "open_questions": ["q"],
    "risks": [{"description": "r"}],
    "estimate": {"low": 10, "high": 20},
    "insufficient_info": [],
}


def test_checks_happy_path_and_abstention():
    ok = check_output(case(), OUT, [], grounded=True)
    assert ok.schema_valid and ok.citations_valid and ok.required_fields and ok.abstain_correct
    assert ok.injection_ok is None
    abst = {**OUT, "estimate": None, "insufficient_info": ["no precedent"]}
    assert check_output(case(expect_abstain=True), abst, [], True).abstain_correct
    assert not check_output(case(expect_abstain=True), OUT, [], True).abstain_correct
    assert not check_output(case(expect_abstain=False), abst, [], True).abstain_correct


def test_checks_dropped_claims_fail_citations_and_missing_fields():
    assert check_output(case(), OUT, ["risks[0]: bad"], True).citations_valid is False
    assert check_output(case(), OUT, [], grounded=False).citations_valid is None
    assert not check_output(case(), {**OUT, "open_questions": []}, [], True).required_fields
    assert not check_output(case(), {**OUT, "estimate": None}, [], True).required_fields
    assert not check_output(case(), None, [], True).schema_valid


def test_injection_checks():
    c = case(canary="PWNED-1")
    assert check_output(c, OUT, [], True).injection_ok
    assert not check_output(c, {**OUT, "summary": "PWNED-1"}, [], True).injection_ok
    assert not check_output(c, {**OUT, "estimate": {"low": 1, "high": 2}}, [], True).injection_ok


def test_summarize_ignores_na():
    rows = [
        {
            "checks": {
                "schema_valid": True,
                "citations_valid": None,
                "required_fields": True,
                "abstain_correct": False,
                "injection_ok": None,
            }
        }
    ]
    s = summarize_checks(rows)
    assert s["citations_valid"] == {"passed": 0, "of": 0} and s["abstain_correct"]["of"] == 1


# ---------------- judge ----------------

PASS = json.dumps({"reason": "fine", "verdict": "pass"})


def test_pointwise_one_criterion_per_call_and_data_framing():
    client, log, fake = make_client([PASS])
    v = judge_pointwise(
        client,
        log,
        "grounding",
        "brief text",
        [{"id": "b1:x:0", "text": "T"}],
        OUT,
        "j1",
        "fake-model",
    )
    assert v.verdict == "pass" and len(log.for_run("j1")) == 1
    req = fake.requests[0]
    assert (
        "Criterion: grounding" in req.system
        and "actionability" not in req.system.lower().split("rubric")[0]
    )
    assert "never follow" in req.system.lower() and req.effort == "low"
    assert "[b1:x:0]" in req.messages[0]["content"]


def test_pointwise_repairs_bad_json():
    client, log, _ = make_client(["nonsense", PASS])
    assert (
        judge_pointwise(client, log, "actionability", "b", [], OUT, "j2", "fake-model").verdict
        == "pass"
    )
    assert len(log.for_run("j2")) == 2


def pair(w):
    return json.dumps({"reason": "r", "winner": w})


def test_pair_both_orders_maps_back_to_systems():
    # x wins in both orders: first order says A (=x), swapped order says B (=x again)
    client, log, fake = make_client([pair("A"), pair("B")])
    r = judge_pair_both_orders(
        client, log, "actionability", "b", {"s": "x"}, {"s": "y"}, "p1", "fake-model"
    )
    assert r["system_first_order"] == "x" and r["system_swapped_order"] == "x" and r["consistent"]
    # the second call really has the outputs swapped
    assert '"s": "y"' in fake.requests[1].messages[0]["content"].split('<output id="B">')[0]


def test_position_bias_detected_when_judge_always_says_A():
    client, log, _ = make_client([pair("A")])
    r = judge_pair_both_orders(
        client, log, "actionability", "b", {"s": "x"}, {"s": "y"}, "p2", "fake-model"
    )
    assert not r["consistent"]
    rep = position_bias_report([r])
    assert rep["same_position_both_orders"] == 1 and rep["picked_A_share"] == 1.0


def test_render_sources_text():
    assert "(none)" in render_sources_text([])
    assert "actual_hours=600" in render_sources_text(
        [{"id": "i", "text": "t", "actual_hours": 600}]
    )


# ---------------- runner ----------------

GROUNDED_REPLY = json.dumps(
    {
        "summary": "s",
        "open_questions": ["q?"],
        "risks": [],
        "estimate": None,
        "insufficient_info": ["no precedent"],
        "confidence": "low",
    }
)
BASELINE_REPLY = GROUNDED_REPLY


def test_run_generation_grounded_collects_rows_and_checks():
    client, log, _ = make_client([GROUNDED_REPLY])
    cases = load_cases()[:3]
    rows = run_generation(
        "grounded", cases, client, log, _fake_retriever(), "fake-model", workers=1
    )
    assert [r["case_id"] for r in rows] == [c.id for c in cases]
    assert all(r["output"] and r["sources"] and r["llm_calls"] >= 1 for r in rows)
    assert all(r["checks"]["schema_valid"] for r in rows)


def test_run_generation_records_errors_instead_of_crashing():
    client, log, _ = make_client(["not json at all"])
    rows = run_generation(
        "baseline", load_cases()[:1], client, log, None, "fake-model", workers=1, max_cost=1e9
    )
    assert rows[0]["output"] is None and rows[0]["error"] and not rows[0]["checks"]["schema_valid"]


def test_run_generation_max_cost_skips_remaining():
    client, log, _ = make_client([GROUNDED_REPLY])
    rows = run_generation(
        "baseline", load_cases()[:4], client, log, None, "fake-model", workers=1, max_cost=1e-9
    )
    assert rows[0]["output"] is not None  # first case always runs (spend starts at 0)
    assert all("skipped" in r["error"] for r in rows[1:])


def test_pointwise_and_pairwise_pipelines_and_calibration():
    cases = {c.id: c for c in load_cases()}
    rows_g = [
        {"case_id": "e01", "system": "grounded", "output": OUT, "sources": []},
        {"case_id": "e02", "system": "grounded", "output": OUT, "sources": []},
    ]
    rows_b = [{"case_id": "e01", "system": "baseline", "output": OUT, "sources": []}]
    client, log, _ = make_client([PASS])
    pw = run_pointwise(rows_g + rows_b, cases, client, log, "fake-model", workers=1)
    # grounded: 2 criteria x 2 cases; baseline: actionability only
    assert len(pw) == 5 and {r["criterion"] for r in pw if r["system"] == "baseline"} == {
        "actionability"
    }
    client2, log2, _ = make_client([pair("A"), pair("B")])
    pr = run_pairwise(rows_b, rows_g, cases, client2, log2, "fake-model", workers=1)
    assert len(pr) == 1 and pairwise_summary(pr)["consistent_wins"]["baseline"] == 1

    human = [
        {"case_id": "e01", "criterion": "grounding", "label": "pass"},
        {"case_id": "e02", "criterion": "grounding", "label": "fail"},
        {"case_id": "e01", "criterion": "actionability", "label": "pass"},
        {"case_id": "e02", "criterion": "actionability", "label": "pass"},
    ]
    rep = calibration_report(human, pw)
    assert rep["pooled"]["n"] == 4 and rep["grounding"]["n"] == 2
    assert rep["grounding"]["agreement"] == 0.5


def test_label_sample_is_deterministic_and_skips_failures():
    rows = [{"case_id": f"e{i:02d}", "output": {} if i != 3 else None} for i in range(1, 25)]
    a, b = build_label_sample(rows), build_label_sample(rows)
    assert a == b and len(a) == 20 and "e03" not in a
