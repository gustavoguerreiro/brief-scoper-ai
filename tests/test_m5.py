import json

import dspy
import pytest

from scoper.dspy_pipeline.lm import ClientLM, run_scope
from scoper.dspy_pipeline.module import GroundedScoper
from scoper.dspy_pipeline.optimize import build_examples, make_metric, program_sha, run_program
from scoper.evals.dataset import load_cases
from scoper.evals.runner import _fake_retriever
from scoper.generation.structured import BudgetExceeded
from scoper.llm.client import CallLog, FakeModel, ModelResponse, build_client
from tests.fakes import OracleModel

PRICES = {"fake-model": {"input": 1.0, "output": 2.0}}


class DspyOracle(OracleModel):
    """Same behaviour as OracleModel, but wrapped in DSPy's chat-adapter output format."""

    def complete(self, req):
        content = "\n".join(m["content"] for m in req.messages)
        body = json.dumps(self._reply(content))
        text = f"[[ ## scope ## ]]\n{body}\n\n[[ ## completed ## ]]"
        return ModelResponse(
            text=text,
            model="fake-model",
            input_tokens=800,
            output_tokens=300,
            stop_reason="end_turn",
        )


class DspyFabricator(DspyOracle):
    fabricate = True


def lm_for(model_obj, budget=None):
    log = CallLog()
    client = build_client(model_obj, log, PRICES, sleep=lambda s: None)
    return ClientLM(client, log, "fake-model", budget_usd=budget), log, client


@pytest.fixture(scope="module")
def retriever():
    return _fake_retriever()


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def example_for(cases, retriever, case_id):
    return build_examples([c for c in cases if c.id == case_id], retriever)[0]


def test_lm_routes_through_our_client_and_logs_cost():
    lm, log, _ = lm_for(FakeModel(["hello"]))
    with run_scope("r1"):
        out = lm(messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    assert out == ["hello"] and len(log.for_run("r1")) == 1 and log.run_cost("r1") > 0
    assert log.for_run("r1")[0].purpose == "dspy"


def test_lm_budget_cap_stops_calls():
    lm, _log, _ = lm_for(FakeModel(["x"]), budget=1e-9)
    lm(prompt="first call is free because spend starts at zero")
    with pytest.raises(BudgetExceeded):
        lm(prompt="second call must be refused")


def test_module_end_to_end_with_a_well_behaved_model(cases, retriever):
    lm, log, _ = lm_for(DspyOracle(cases))
    ex = example_for(cases, retriever, "e01")
    with dspy.context(lm=lm), run_scope("m1"):
        pred = GroundedScoper()(brief=ex.brief, hits=ex.hits)
    assert pred.dropped == [] and pred.scope.estimate is not None
    assert pred.scope.estimate.citations and len(log.for_run("m1")) == 1  # no repair needed


def test_module_repairs_then_drops_unverifiable_claims(cases, retriever):
    lm, log, _ = lm_for(DspyFabricator(cases))
    ex = example_for(cases, retriever, "e01")
    with dspy.context(lm=lm), run_scope("m2"):
        pred = GroundedScoper(max_repairs=2)(brief=ex.brief, hits=ex.hits)
    assert len(log.for_run("m2")) == 3  # first try + 2 repairs
    assert pred.dropped and pred.scope.estimate is None and pred.scope.confidence == "low"


def test_run_program_rows_match_the_eval_pipeline_format(cases, retriever):
    lm, log, _ = lm_for(DspyOracle(cases))
    exs = build_examples([c for c in cases if c.id in ("e01", "e20")], retriever)
    by_id = {c.id: c for c in cases}
    with dspy.context(lm=lm):
        rows = run_program("baseline", GroundedScoper(), exs, by_id, log, workers=1)
    assert [r["case_id"] for r in rows] == ["e01", "e20"]
    assert rows[0]["checks"]["abstain_correct"] and rows[1]["checks"]["abstain_correct"]
    assert rows[0]["prompts"]["scope_grounded_dspy"].startswith("scope_grounded_dspy@v1#")
    assert rows[0]["program"].startswith("baseline#") and rows[0]["sources"]
    assert rows[0]["llm_calls"] == 1 and rows[0]["cost_usd"] > 0


def test_metric_scores_code_checks_and_judge_separately(cases, retriever):
    lm, _log, _ = lm_for(DspyOracle(cases))
    ex = example_for(cases, retriever, "e01")
    with dspy.context(lm=lm):
        pred = GroundedScoper()(brief=ex.brief, hits=ex.hits)
    jlog = CallLog()
    verdict = lambda v: build_client(
        FakeModel([json.dumps({"reason": "r", "verdict": v})]), jlog, PRICES, sleep=lambda s: None
    )
    by_id = {c.id: c for c in cases}
    assert make_metric(by_id, verdict("pass"), jlog, "fake-model")(ex, pred) == 1.0
    assert make_metric(by_id, verdict("fail"), jlog, "fake-model")(ex, pred) == 0.5


def test_bootstrap_optimizer_collects_demos_offline(cases, retriever):
    lm, _log, _ = lm_for(DspyOracle(cases))
    ids = ["e01", "e02", "e03"]
    train = build_examples([c for c in cases if c.id in ids], retriever)
    jlog = CallLog()
    judge = build_client(
        FakeModel([json.dumps({"reason": "r", "verdict": "pass"})]),
        jlog,
        PRICES,
        sleep=lambda s: None,
    )
    metric = make_metric({c.id: c for c in cases}, judge, jlog, "fake-model")
    opt = dspy.teleprompt.BootstrapFewShot(
        metric=metric, metric_threshold=1.0, max_bootstrapped_demos=2, max_labeled_demos=0
    )
    with dspy.context(lm=lm):
        tuned = opt.compile(GroundedScoper(), trainset=train)
    assert 1 <= len(tuned.predict.demos) <= 2
    assert program_sha(tuned) != program_sha(GroundedScoper())  # the program really changed


def test_failed_generation_is_recorded_not_raised(cases, retriever):
    lm, log, _ = lm_for(FakeModel(["this is not in the adapter format at all"]))
    exs = build_examples([c for c in cases if c.id == "e01"], retriever)
    with dspy.context(lm=lm):
        rows = run_program(
            "broken", GroundedScoper(), exs, {c.id: c for c in cases}, log, workers=1
        )
    assert rows[0]["output"] is None and rows[0]["error"] and not rows[0]["checks"]["schema_valid"]
