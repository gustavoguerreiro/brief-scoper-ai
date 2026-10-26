"""Eval orchestration and CLI.

uv run python -m scoper.evals.runner retrieval [--fake]
uv run python -m scoper.evals.runner generate --system grounded|baseline [--split all] [--max-cost 2]
uv run python -m scoper.evals.runner report
uv run python -m scoper.evals.runner judge-pointwise [--max-cost 3]
uv run python -m scoper.evals.runner judge-pairwise  [--max-cost 3]
uv run python -m scoper.evals.runner make-sample
uv run python -m scoper.evals.label           # you label the sample by hand
uv run python -m scoper.evals.runner calibrate
"""

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scoper.evals.checks import check_output, checks_dict, summarize_checks
from scoper.evals.dataset import (
    CASES_PATH,
    HARD_PATH,
    LABELS_DIR,
    RUNS_DIR,
    EvalCase,
    load_cases,
    read_jsonl,
    write_jsonl,
)
from scoper.evals.judge import (
    judge_pair_both_orders,
    judge_pointwise,
    position_bias_report,
)
from scoper.evals.metrics import bootstrap_ci, cohen_kappa, confusion, percent_agreement
from scoper.generation.cited import generate_cited
from scoper.generation.scope import generate_scope
from scoper.generation.structured import BudgetExceeded, ScopeGenerationError
from scoper.llm.client import CallLog, ModelClient, PermanentModelError, TransientModelError
from scoper.retrieval.hybrid import Retriever

MODEL_ERRORS = (ScopeGenerationError, BudgetExceeded, TransientModelError, PermanentModelError)
CRITERIA = ("grounding", "actionability")


def _spent(log: CallLog) -> float:
    return sum(r.cost_usd for r in list(log.records))


def _source_dict(chunk) -> dict:
    m = chunk.meta
    return {
        "id": chunk.id,
        "brief_id": chunk.brief_id,
        "section": chunk.section,
        "text": chunk.text,
        "title": m.get("title"),
        "outcome": m.get("outcome"),
        "actual_hours": m.get("actual_hours"),
        "actual_cost": m.get("actual_cost"),
    }


def run_generation(
    system: str,
    cases: list[EvalCase],
    client: ModelClient,
    log: CallLog,
    retriever: Retriever | None,
    model: str,
    workers: int = 4,
    max_cost: float | None = None,
    top_k: int = 5,
    per_brief_cap: int | None = None,
) -> list[dict]:
    if system == "grounded" and retriever is None:
        raise ValueError("grounded generation needs a retriever")
    start_spend = _spent(log)

    def one(case: EvalCase) -> dict:
        run_id = f"{system}-{case.id}"
        row = {
            "case_id": case.id,
            "system": system,
            "split": case.split,
            "tags": case.tags,
            "output": None,
            "error": "",
            "dropped": [],
            "sources": [],
        }
        t0 = time.perf_counter()
        if max_cost is not None and _spent(log) - start_spend >= max_cost:
            row["error"] = "skipped: max_cost reached"
        else:
            try:
                if system == "grounded":
                    hits = retriever.search(
                        case.brief, k=top_k, run_id=run_id, per_brief_cap=per_brief_cap
                    )
                    res = generate_cited(client, log, case.brief, hits, run_id, model)
                    row.update(
                        output=res.output.model_dump(),
                        dropped=res.dropped,
                        sources=[_source_dict(h.chunk) for h in hits],
                    )
                else:
                    row["output"] = generate_scope(
                        client, log, case.brief, run_id, model
                    ).model_dump()
            except MODEL_ERRORS as e:
                row["error"] = repr(e)[:300]
        row.update(
            cost_usd=round(log.run_cost(run_id), 6),
            llm_calls=len(log.for_run(run_id)),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        row["checks"] = checks_dict(
            check_output(case, row["output"], row["dropped"], grounded=system == "grounded")
        )
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, cases))


def run_pointwise(
    rows: list[dict],
    cases: dict[str, EvalCase],
    client: ModelClient,
    log: CallLog,
    model: str,
    workers: int = 4,
    max_cost: float | None = None,
) -> list[dict]:
    start_spend = _spent(log)
    jobs = [
        (r, crit)
        for r in rows
        if r["output"] is not None
        for crit in CRITERIA
        if not (crit == "grounding" and r["system"] != "grounded")  # baseline has no sources
    ]

    def one(job) -> dict:
        r, crit = job
        out = {"case_id": r["case_id"], "system": r["system"], "criterion": crit}
        if max_cost is not None and _spent(log) - start_spend >= max_cost:
            return {**out, "verdict": None, "reason": "skipped: max_cost reached"}
        try:
            v = judge_pointwise(
                client,
                log,
                crit,
                cases[r["case_id"]].brief,
                r["sources"],
                r["output"],
                f"jp-{r['system']}-{r['case_id']}-{crit}",
                model,
            )
            return {**out, "verdict": v.verdict, "reason": v.reason}
        except MODEL_ERRORS as e:
            return {**out, "verdict": None, "reason": f"error: {e!r}"[:300]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def run_pairwise(
    baseline: list[dict],
    grounded: list[dict],
    cases: dict[str, EvalCase],
    client: ModelClient,
    log: CallLog,
    model: str,
    workers: int = 4,
    max_cost: float | None = None,
) -> list[dict]:
    """x = baseline, y = grounded; criterion actionability (the only one fair to both)."""
    start_spend = _spent(log)
    by_id = {r["case_id"]: r for r in baseline if r["output"] is not None}
    jobs = [
        (by_id[g["case_id"]], g)
        for g in grounded
        if g["output"] is not None and g["case_id"] in by_id
    ]

    def one(job) -> dict:
        b, g = job
        cid = b["case_id"]
        if max_cost is not None and _spent(log) - start_spend >= max_cost:
            return {"case_id": cid, "criterion": "actionability", "skipped": True}
        try:
            res = judge_pair_both_orders(
                client,
                log,
                "actionability",
                cases[cid].brief,
                b["output"],
                g["output"],
                f"jw-{cid}",
                model,
            )
        except MODEL_ERRORS as e:
            return {
                "case_id": cid,
                "criterion": "actionability",
                "skipped": True,
                "error": repr(e)[:300],
            }
        return {"case_id": cid, "criterion": "actionability", **res}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def pairwise_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("skipped")]
    both_decisive = [r for r in ok if r["consistent"] and r["system_first_order"] != "tie"]
    return {
        "x=baseline,y=grounded": True,
        "position_bias": position_bias_report(ok),
        "consistent_wins": {
            "baseline": sum(r["system_first_order"] == "x" for r in both_decisive),
            "grounded": sum(r["system_first_order"] == "y" for r in both_decisive),
            "consistent_ties": sum(
                r["consistent"] and r["system_first_order"] == "tie" for r in ok
            ),
            "inconsistent": sum(not r["consistent"] for r in ok),
        },
        "skipped": len(rows) - len(ok),
    }


def _num(x: float) -> float | None:
    """JSON-friendly: NaN (undefined statistic) becomes null."""
    return None if math.isnan(x) else round(x, 3)


def build_label_sample(grounded_rows: list[dict], n: int = 20, seed: int = 7) -> list[str]:
    ids = sorted(r["case_id"] for r in grounded_rows if r["output"] is not None)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def calibration_report(human: list[dict], judge: list[dict], system: str = "grounded") -> dict:
    """Join on (case_id, criterion). Labels are 'pass'/'fail'."""
    jmap = {
        (r["case_id"], r["criterion"]): r["verdict"]
        for r in judge
        if r["system"] == system and r["verdict"]
    }
    pairs = [
        (h["label"], jmap[(h["case_id"], h["criterion"])], h["criterion"])
        for h in human
        if (h["case_id"], h["criterion"]) in jmap
    ]

    def stats(sel: list[tuple]) -> dict:
        if not sel:
            return {"n": 0}
        a, b = [p[0] for p in sel], [p[1] for p in sel]
        k = cohen_kappa(a, b)
        lo, hi = bootstrap_ci(a, b)
        return {
            "n": len(sel),
            "agreement": round(percent_agreement(a, b), 3),
            "kappa": _num(k),
            "kappa_95ci": [_num(lo), _num(hi)],
            "human_pass_rate": round(a.count("pass") / len(a), 3),
            "judge_pass_rate": round(b.count("pass") / len(b), 3),
            "confusion": confusion(a, b),
        }

    return {"pooled": stats(pairs), **{c: stats([p for p in pairs if p[2] == c]) for c in CRITERIA}}


# ----------------------------- CLI -----------------------------


def _runtime(need_retriever: bool = True):
    from dotenv import load_dotenv

    from scoper.llm.client import AnthropicClient, build_client

    load_dotenv()
    log = CallLog(path=Path("logs/llm_calls.jsonl"))
    client = build_client(AnthropicClient(), log)
    retriever = None
    if need_retriever:
        from scoper.evals.retrieval_eval import CachingEmbedder
        from scoper.llm.embeddings import OpenAIEmbedder
        from scoper.retrieval.pg_store import PgStore, database_url

        emb = CachingEmbedder(OpenAIEmbedder(log))
        retriever = Retriever(PgStore(database_url(), emb.dim), emb)
    return log, client, retriever


def _fake_retriever() -> Retriever:
    from scoper.ingestion.importer import import_briefs, load_dir
    from scoper.llm.embeddings import FakeEmbedder
    from scoper.retrieval.store import InMemoryStore

    store, emb = InMemoryStore(), FakeEmbedder()
    import_briefs(load_dir(Path("data/briefs")), store, emb)
    return Retriever(store, emb)


def _cmd_retrieval(args) -> None:
    from scoper.evals.retrieval_eval import evaluate_retrieval, format_report

    retriever = _fake_retriever() if args.fake else _runtime()[2]
    path = HARD_PATH if args.set == "hard" else CASES_PATH
    report = evaluate_retrieval(retriever, load_cases(path, split=args.split))
    print(format_report(report))
    name = ("retrieval_fake" if args.fake else "retrieval") + (
        "_hard" if args.set == "hard" else ""
    )
    path = RUNS_DIR / f"{name}_{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"saved {path}")


def _cmd_generate(args) -> None:
    import os

    model = os.getenv("SCOPER_MODEL", "claude-sonnet-5-5")
    log, client, retriever = _runtime(need_retriever=args.system == "grounded")
    cases = load_cases(split=args.split)
    t0 = time.perf_counter()
    rows = run_generation(
        args.system,
        cases,
        client,
        log,
        retriever,
        model,
        workers=args.workers,
        max_cost=args.max_cost,
        per_brief_cap=args.per_brief_cap,
    )
    path = RUNS_DIR / f"{args.system}_{args.split}.jsonl"
    write_jsonl(path, rows)
    spent = sum(r["cost_usd"] for r in rows)
    print(f"{len(rows)} cases, cost ${spent:.4f}, wall {time.perf_counter() - t0:.0f}s -> {path}")
    print(json.dumps(summarize_checks(rows), indent=1))


def _cmd_report(args) -> None:
    for p in sorted(RUNS_DIR.glob("*_*.jsonl")):
        rows = read_jsonl(p)
        if rows and "checks" in rows[0]:
            lat = sorted(r["latency_ms"] for r in rows)
            print(
                p.name,
                f"cost ${sum(r['cost_usd'] for r in rows):.4f}",
                f"median latency {lat[len(lat) // 2]:.0f} ms",
            )
            print(json.dumps(summarize_checks(rows)))


def _cmd_judge_pointwise(args) -> None:
    import os

    model = os.getenv("SCOPER_JUDGE_MODEL", "claude-opus-5-5")
    log, client, _ = _runtime(need_retriever=False)
    cases = {c.id: c for c in load_cases()}
    rows = []
    for system in ("grounded", "baseline"):
        rows += read_jsonl(RUNS_DIR / f"{system}_all.jsonl")
    out = run_pointwise(
        rows, cases, client, log, model, workers=args.workers, max_cost=args.max_cost
    )
    write_jsonl(RUNS_DIR / "judge_pointwise.jsonl", out)
    done = [r for r in out if r["verdict"]]
    print(f"{len(done)}/{len(out)} judged with {model}, cost ${_spent(log):.4f}")
    for system in ("grounded", "baseline"):
        for crit in CRITERIA:
            sel = [r for r in done if r["system"] == system and r["criterion"] == crit]
            if sel:
                print(
                    f"{system:9} {crit:14} pass {sum(r['verdict'] == 'pass' for r in sel)}/{len(sel)}"
                )


def _cmd_judge_pairwise(args) -> None:
    import os

    model = os.getenv("SCOPER_JUDGE_MODEL", "claude-opus-5-5")
    log, client, _ = _runtime(need_retriever=False)
    cases = {c.id: c for c in load_cases()}
    out = run_pairwise(
        read_jsonl(RUNS_DIR / "baseline_all.jsonl"),
        read_jsonl(RUNS_DIR / "grounded_all.jsonl"),
        cases,
        client,
        log,
        model,
        workers=args.workers,
        max_cost=args.max_cost,
    )
    write_jsonl(RUNS_DIR / "judge_pairwise.jsonl", out)
    print(f"judge {model}, cost ${_spent(log):.4f}")
    print(json.dumps(pairwise_summary(out), indent=1))


def _cmd_make_sample(args) -> None:
    ids = build_label_sample(read_jsonl(RUNS_DIR / "grounded_all.jsonl"))
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / "sample.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"{len(ids)} cases to label (x{len(CRITERIA)} criteria): {ids}")


def _cmd_calibrate(args) -> None:
    human = read_jsonl(LABELS_DIR / "human.jsonl")
    judge = read_jsonl(RUNS_DIR / "judge_pointwise.jsonl")
    print(json.dumps(calibration_report(human, judge), indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--fake", action="store_true")
    r.add_argument("--split", default="all")
    r.add_argument("--set", choices=["main", "hard"], default="main")
    r.set_defaults(fn=_cmd_retrieval)
    g = sub.add_parser("generate")
    g.add_argument("--system", choices=["grounded", "baseline"], required=True)
    g.add_argument("--split", default="all")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--max-cost", type=float, default=2.0)
    g.add_argument("--per-brief-cap", type=int, default=None)
    g.set_defaults(fn=_cmd_generate)
    sub.add_parser("report").set_defaults(fn=_cmd_report)
    for name, fn in (
        ("judge-pointwise", _cmd_judge_pointwise),
        ("judge-pairwise", _cmd_judge_pairwise),
    ):
        j = sub.add_parser(name)
        j.add_argument("--workers", type=int, default=4)
        j.add_argument("--max-cost", type=float, default=3.0)
        j.set_defaults(fn=fn)
    sub.add_parser("make-sample").set_defaults(fn=_cmd_make_sample)
    sub.add_parser("calibrate").set_defaults(fn=_cmd_calibrate)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
