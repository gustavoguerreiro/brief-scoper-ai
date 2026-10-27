"""Optimize and evaluate the DSPy pipeline.

    uv run python -m scoper.dspy_pipeline.optimize eval     --name baseline --split test
    uv run python -m scoper.dspy_pipeline.optimize optimize --budget 0.5
    uv run python -m scoper.dspy_pipeline.optimize eval     --name optimized_v1 --program prompts/dspy/optimized_v1.json --split test

The optimization metric is cheap on purpose: code checks AND a grounding verdict from a small judge
model. The held-out comparison then re-judges outputs with the stronger M3 judge (`runner judge-pointwise`
style), so the optimizer is never graded by the model it optimized against.
"""

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import dspy

from scoper.dspy_pipeline.lm import ClientLM, run_scope
from scoper.dspy_pipeline.module import GroundedScoper
from scoper.evals.checks import check_output, checks_dict
from scoper.evals.dataset import RUNS_DIR, EvalCase, load_cases, write_jsonl
from scoper.evals.judge import judge_pointwise
from scoper.evals.runner import MODEL_ERRORS, _source_dict
from scoper.llm.client import CallLog, ModelClient
from scoper.prompts.registry import prompt_ids

DSPY_DIR = Path(__file__).resolve().parents[3] / "prompts" / "dspy"


def build_examples(cases: list[EvalCase], retriever, top_k: int = 5) -> list[dspy.Example]:
    out = []
    for c in cases:
        hits = retriever.search(c.brief, k=top_k, run_id=f"dspy-ret-{c.id}")
        out.append(
            dspy.Example(case_id=c.id, brief=c.brief, hits=hits).with_inputs("brief", "hits")
        )
    return out


def _code_ok(case: EvalCase, scope: dict, dropped: list[str]) -> bool:
    checks = check_output(case, scope, dropped, grounded=True)
    return all(v for v in asdict(checks).values() if v is not None)


def make_metric(
    cases: dict[str, EvalCase], client: ModelClient, log: CallLog, judge_model: str
) -> Callable:
    """Score in {0, 0.5, 1}: half for passing every code check, half for a grounding pass."""

    def metric(example, pred, trace=None) -> float:
        case = cases[example.case_id]
        scope = pred.scope.model_dump()
        code = _code_ok(case, scope, pred.dropped)
        sources = [_source_dict(h.chunk) for h in example.hits]
        verdict = judge_pointwise(
            client,
            log,
            "grounding",
            case.brief,
            sources,
            scope,
            f"metric-{example.case_id}",
            judge_model,
        )
        return (float(code) + float(verdict.verdict == "pass")) / 2

    return metric


def program_sha(program: dspy.Module) -> str:
    return hashlib.sha256(
        json.dumps(program.dump_state(), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def run_program(
    name: str,
    program: dspy.Module,
    examples: list[dspy.Example],
    cases: dict[str, EvalCase],
    log: CallLog,
    workers: int = 3,
) -> list[dict]:
    """Run a DSPy program over examples and return rows in the same format as `runner generate`."""
    sha = program_sha(program)
    lm = dspy.settings.lm  # context-managed LMs are thread-local: hand it to the workers explicitly

    def one(ex: dspy.Example) -> dict:
        case = cases[ex.case_id]
        run_id = f"dspy-{name}-{ex.case_id}"
        row = {
            "case_id": ex.case_id,
            "system": "dspy_grounded",
            "split": case.split,
            "tags": case.tags,
            "output": None,
            "error": "",
            "dropped": [],
            "sources": [_source_dict(h.chunk) for h in ex.hits],
            "prompts": prompt_ids("scope_grounded_dspy"),
            "program": f"{name}#{sha}",
        }
        t0 = time.perf_counter()
        try:
            with dspy.context(lm=lm), run_scope(run_id):
                pred = program(brief=ex.brief, hits=ex.hits)
            row["output"], row["dropped"] = pred.scope.model_dump(), pred.dropped
        except MODEL_ERRORS + (
            Exception,
        ) as e:  # DSPy parse errors are not ours: record, don't crash
            row["error"] = repr(e)[:300]
        row.update(
            cost_usd=round(log.run_cost(run_id), 6),
            llm_calls=len(log.for_run(run_id)),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        row["checks"] = checks_dict(check_output(case, row["output"], row["dropped"], True))
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, examples))


# ----------------------------- CLI -----------------------------


def _setup(args):
    import os

    from dotenv import load_dotenv

    from scoper.evals.retrieval_eval import CachingEmbedder
    from scoper.llm.client import AnthropicClient, build_client
    from scoper.llm.embeddings import OpenAIEmbedder
    from scoper.retrieval.hybrid import Retriever
    from scoper.retrieval.pg_store import PgStore, database_url

    load_dotenv()
    log = CallLog(path=Path("logs/llm_calls.jsonl"))
    client = build_client(AnthropicClient(), log)
    emb = CachingEmbedder(OpenAIEmbedder(log))
    retriever = Retriever(PgStore(database_url(), emb.dim), emb)
    model = os.getenv("SCOPER_MODEL", "claude-sonnet-5-5")
    dspy.configure(lm=ClientLM(client, log, model, budget_usd=args.budget))
    return log, client, retriever


def _load_program(path: str | None) -> GroundedScoper:
    program = GroundedScoper()
    if path:
        program.load(path)
    return program


def _cmd_eval(args) -> None:
    log, _client, retriever = _setup(args)
    cases = {c.id: c for c in load_cases()}
    examples = build_examples(load_cases(split=args.split), retriever)
    rows = run_program(args.name, _load_program(args.program), examples, cases, log, args.workers)
    path = RUNS_DIR / f"dspy_{args.name}_{args.split}.jsonl"
    write_jsonl(path, rows)
    cost = sum(r["cost_usd"] for r in rows)
    ok = sum(r["checks"]["schema_valid"] for r in rows)
    print(
        f"{len(rows)} cases, schema valid {ok}, generation cost ${cost:.4f}, total spend ${sum(r.cost_usd for r in log.records):.4f} -> {path}"
    )


def _cmd_optimize(args) -> None:
    import os

    log, client, retriever = _setup(args)
    cases = {c.id: c for c in load_cases()}
    train = build_examples(load_cases(split="dev"), retriever)  # dev only: test stays untouched
    judge_model = os.getenv("SCOPER_METRIC_JUDGE", "claude-haiku-5-5")
    metric = make_metric(cases, client, log, judge_model)
    opt = dspy.teleprompt.BootstrapFewShot(
        metric=metric,
        metric_threshold=1.0,
        max_bootstrapped_demos=args.demos,
        max_labeled_demos=0,
        max_rounds=1,
        max_errors=args.max_errors,
    )
    t0 = time.perf_counter()
    tuned = opt.compile(GroundedScoper(), trainset=train)
    DSPY_DIR.mkdir(parents=True, exist_ok=True)
    out = DSPY_DIR / f"{args.name}.json"
    tuned.save(str(out))
    demos = len(tuned.predict.demos)
    print(
        f"optimized in {time.perf_counter() - t0:.0f}s, {demos} demos kept, spend ${sum(r.cost_usd for r in log.records):.4f} -> {out}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--name", required=True)
    e.add_argument("--program", default=None, help="saved optimized program (default: unoptimized)")
    e.add_argument("--split", choices=["dev", "test", "all"], default="test")
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--budget", type=float, default=0.6)
    e.set_defaults(fn=_cmd_eval)
    o = sub.add_parser("optimize")
    o.add_argument("--name", default="optimized_v1")
    o.add_argument("--demos", type=int, default=3)
    o.add_argument("--max-errors", type=int, default=3)
    o.add_argument("--budget", type=float, default=0.6)
    o.set_defaults(fn=_cmd_optimize)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
