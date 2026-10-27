import json

import pytest

from scoper.evals import gate
from scoper.evals.dataset import load_cases
from scoper.evals.runner import _fake_retriever, run_generation
from scoper.llm.client import CallLog, build_client
from scoper.prompts import registry
from scoper.prompts.registry import Prompt, PromptError, get_prompt, list_versions, prompt_ids
from tests.fakes import FabricatingModel, OracleModel

PRICES = {"fake-model": {"input": 1.0, "output": 2.0}}
FLOORS = gate.load_thresholds()["generation"]["grounded"]


# ---------------- prompt registry ----------------


def test_every_active_prompt_exists_and_renders():
    for name, version in registry.active_versions().items():
        p = get_prompt(name)
        assert p.version == version and version in list_versions(name)
        values = {ph: "x" for ph in p.placeholders}
        assert "<<" not in p.render(**values)


def test_prompt_id_changes_when_text_changes():
    a, b = Prompt("n", "v1", "hello <<x>>"), Prompt("n", "v1", "hello there <<x>>")
    assert a.id != b.id and a.id.startswith("n@v1#")
    assert a.render(x="1") == "hello 1"


def test_render_rejects_missing_or_unexpected_values():
    p = Prompt("n", "v1", "a <<x>> b")
    with pytest.raises(PromptError):
        p.render()
    with pytest.raises(PromptError):
        p.render(x="1", y="2")


def test_registry_reads_dir_active_pointer_and_env_override(tmp_path, monkeypatch):
    d = tmp_path / "demo"
    d.mkdir()
    for v in ("v1", "v2", "v10"):
        (d / f"{v}.md").write_text(
            f"---\nversion: {v}\nnotes: n\n---\ntext {v}\n", encoding="utf-8"
        )
    (tmp_path / "active.json").write_text(json.dumps({"demo": "v1"}), encoding="utf-8")
    monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
    monkeypatch.setattr(registry, "ACTIVE_PATH", tmp_path / "active.json")
    assert list_versions("demo") == ["v1", "v2", "v10"]  # numeric, not alphabetic, order
    assert get_prompt("demo").text == "text v1"
    assert get_prompt("demo", "v2").text == "text v2"
    monkeypatch.setenv("SCOPER_PROMPT_DEMO", "v10")
    assert get_prompt("demo").version == "v10"
    with pytest.raises(PromptError):
        get_prompt("demo", "v99")
    with pytest.raises(PromptError):
        get_prompt("nope")


def test_front_matter_must_match_file_name(tmp_path, monkeypatch):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "v1.md").write_text("---\nversion: v2\n---\nx\n", encoding="utf-8")
    monkeypatch.setattr(registry, "PROMPTS_DIR", tmp_path)
    with pytest.raises(PromptError, match="does not match"):
        get_prompt("demo", "v1")


def test_prompts_are_loaded_from_registry_by_generation(monkeypatch):
    from scoper.generation.scope import generate_scope
    from scoper.llm.client import FakeModel

    reply = json.dumps(
        {
            "summary": "s",
            "open_questions": ["q"],
            "risks": [],
            "estimate": None,
            "insufficient_info": ["x"],
            "confidence": "low",
        }
    )
    fake, log = FakeModel([reply]), CallLog()
    client = build_client(fake, log, PRICES, sleep=lambda s: None)
    custom = Prompt("scope_baseline", "v9", "CUSTOM PROMPT <<schema>>")
    generate_scope(client, log, "a brief of enough length", "r1", "fake-model", prompt=custom)
    assert fake.requests[0].system.startswith("CUSTOM PROMPT")


# ---------------- gate rules ----------------


def test_floors_pass_fail_warn_and_missing():
    f = gate.check_floors(
        "x", {"a": 0.9, "b": 0.5, "c": None}, {"a": 0.8, "b": 0.8, "c": 0.5, "d": None}
    )
    levels = {x.where: x.level for x in f}
    assert levels == {"x.a": "ok", "x.b": "FAIL", "x.c": "FAIL", "x.d": "WARN"}
    assert gate.exit_code(f) == 1


def test_skipped_checks_do_not_fail_unless_strict():
    f = gate.check_floors("x", {}, {"a": None})
    assert gate.exit_code(f) == 0 and gate.exit_code(f, strict=True) == 1


def test_regression_against_baseline():
    base = {"schema_valid": 1.0, "citations_valid": 1.0}
    assert (
        gate.check_regression("g", {"schema_valid": 0.96, "citations_valid": 1.0}, base, 0.05) == []
    )
    bad = gate.check_regression("g", {"schema_valid": 0.90, "citations_valid": 1.0}, base, 0.05)
    assert [x.where for x in bad] == ["g.schema_valid"]


def test_thresholds_file_is_consistent():
    th = gate.load_thresholds()
    assert 0 < th["max_regression"] < 0.5
    for section in ("retrieval_fake", "retrieval_real"):
        for floors in th[section].values():
            for metrics in floors.values():
                assert all(0 <= v <= 1 for v in metrics.values())


def test_fake_retrieval_gate_passes_on_current_code():
    from scoper.evals.dataset import CASES_PATH, HARD_PATH
    from scoper.evals.retrieval_eval import evaluate_retrieval

    th, retriever, found = gate.load_thresholds()["retrieval_fake"], _fake_retriever(), []
    for label, path in (("main", CASES_PATH), ("hard", HARD_PATH)):
        rep = evaluate_retrieval(retriever, load_cases(path))
        found += gate.retrieval_findings(rep, th[label], label)
    assert found and gate.exit_code(found) == 0


def test_retrieval_gate_fails_when_a_metric_collapses():
    report = {"configs": {"hybrid": {"hit@3": 0.40, "mrr": 0.30}}}
    f = gate.retrieval_findings(report, {"hybrid": {"hit@3": 0.95}, "bm25": {"mrr": 0.9}}, "r")
    assert {x.where: x.level for x in f} == {"r.hybrid.hit@3": "FAIL", "r.bm25": "FAIL"}


# ---------------- the pipeline gate must be able to pass AND fail ----------------


def _grounded_rows(model_cls):
    cases = load_cases()
    log = CallLog()
    client = build_client(model_cls(cases), log, PRICES, sleep=lambda s: None)
    return run_generation(
        "grounded", cases, client, log, _fake_retriever(), "fake-model", workers=1
    )


def test_gate_passes_for_a_well_behaved_generator():
    rows = _grounded_rows(OracleModel)
    found = gate.generation_findings(rows, FLOORS, "oracle")
    assert gate.exit_code(found) == 0, gate.render(found)
    assert all(r["prompts"]["scope_grounded"].startswith("scope_grounded@v1#") for r in rows)


def test_gate_fails_for_a_generator_that_fabricates_citations():
    rows = _grounded_rows(FabricatingModel)
    found = gate.generation_findings(rows, FLOORS, "fab")
    failed = {f.where for f in found if f.level == "FAIL"}
    assert "fab.citations_valid" in failed and gate.exit_code(found) == 1


def test_gate_flags_a_regression_even_when_floors_still_pass():
    rows = _grounded_rows(OracleModel)
    baseline = {k: 1.0 for k in FLOORS}
    broken = [dict(r, checks=dict(r["checks"], required_fields=False)) for r in rows[:2]] + rows[2:]
    found = gate.generation_findings(broken, {"required_fields": 0.5}, "g", baseline, 0.05)
    assert any(f.level == "FAIL" and "dropped" in f.message for f in found)


def test_prompt_ids_helper():
    ids = prompt_ids("scope_grounded", "judge_pointwise")
    assert set(ids) == {"scope_grounded", "judge_pointwise"}


# ---------------- immutability and freshness ----------------


def test_frozen_prompt_versions_are_immutable():
    hashes = registry.recorded_hashes()
    for name, version in registry.all_versions():
        assert f"{name}/{version}" in hashes, (
            f"run `python -m scoper.prompts.registry` to freeze {name}/{version}"
        )
        assert get_prompt(name, version).sha == hashes[f"{name}/{version}"], (
            f"{name}/{version} was edited after being frozen: create a new version instead"
        )


def test_gate_rejects_results_produced_by_other_prompt_versions():
    rows = _grounded_rows(OracleModel)
    assert gate.exit_code(gate.prompt_freshness(rows, "x")) == 0
    stale = [dict(r, prompts={"scope_grounded": "scope_grounded@v1#000000000000"}) for r in rows]
    found = gate.prompt_freshness(stale, "x")
    assert found[0].level == "FAIL" and "stale" in found[0].message
    untracked = [{k: v for k, v in r.items() if k != "prompts"} for r in rows]
    assert gate.prompt_freshness(untracked, "x")[0].level == "FAIL"
