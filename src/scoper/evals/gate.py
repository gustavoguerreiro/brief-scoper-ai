"""Quality gate: turn eval results into a pass/fail exit code for CI.

    uv run python -m scoper.evals.gate retrieval --fake            # offline, free, runs in CI
    uv run python -m scoper.evals.gate retrieval                   # real embeddings (needs DB + key)
    uv run python -m scoper.evals.gate generation --split dev      # checks stored by `runner generate`
    uv run python -m scoper.evals.gate judge                       # judge pass rates (once calibrated)

Two kinds of rules, both in evals/thresholds.json:
- floors: an absolute minimum per metric. These numbers were CHOSEN by me with a margin below what I
  measured; they are not measurements and they are not statistically derived.
- regression: a metric may not drop more than `max_regression` below a saved baseline file.
A floor of null means "not calibrated yet": the gate reports a warning and skips it, never a pass.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from scoper.evals.checks import summarize_checks
from scoper.evals.dataset import CASES_PATH, HARD_PATH, ROOT, RUNS_DIR, load_cases, read_jsonl

THRESHOLDS_PATH = ROOT / "evals" / "thresholds.json"
BASELINES_DIR = ROOT / "evals" / "baselines"


@dataclass(frozen=True)
class Finding:
    level: str  # "FAIL" | "WARN" | "ok"
    where: str
    message: str


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_floors(where: str, metrics: dict[str, float | None], floors: dict) -> list[Finding]:
    out = []
    for name, floor in floors.items():
        value = metrics.get(name)
        if floor is None:
            out.append(Finding("WARN", f"{where}.{name}", "no floor set (not calibrated): skipped"))
        elif value is None:
            out.append(
                Finding("FAIL", f"{where}.{name}", "metric missing or had no applicable cases")
            )
        elif value < floor:
            out.append(Finding("FAIL", f"{where}.{name}", f"{value:.3f} < floor {floor:.3f}"))
        else:
            out.append(Finding("ok", f"{where}.{name}", f"{value:.3f} >= floor {floor:.3f}"))
    return out


def check_regression(
    where: str, current: dict[str, float | None], baseline: dict[str, float | None], tol: float
) -> list[Finding]:
    out = []
    for name, base in baseline.items():
        cur = current.get(name)
        if base is None or cur is None:
            continue
        if base - cur > tol + 1e-9:
            out.append(
                Finding("FAIL", f"{where}.{name}", f"dropped {base:.3f} -> {cur:.3f} (> {tol})")
            )
    return out


def retrieval_findings(report: dict, floors: dict, label: str) -> list[Finding]:
    out = []
    for config, metric_floors in floors.items():
        got = report["configs"].get(config)
        if got is None:
            out.append(Finding("FAIL", f"{label}.{config}", "config missing from report"))
            continue
        out += check_floors(f"{label}.{config}", got, metric_floors)
    return out


def check_rates(rows: list[dict]) -> dict[str, float | None]:
    summary = summarize_checks(rows)
    return {
        k: (v["passed"] / v["of"] if v["of"] else None)
        for k, v in summary.items()
        if isinstance(v, dict)
    }


def prompt_freshness(rows: list[dict], label: str) -> list[Finding]:
    """Results only count for the prompts that produced them: stale or untracked runs fail."""
    from scoper.prompts.registry import get_prompt

    seen: dict[str, set[str]] = {}
    for r in rows:
        for name, pid in (r.get("prompts") or {}).items():
            seen.setdefault(name, set()).add(pid)
    if not seen:
        return [
            Finding("FAIL", f"{label}.prompts", "run does not record prompt versions: rerun it")
        ]
    out = []
    for name, ids in seen.items():
        active = get_prompt(name).id
        if ids != {active}:
            out.append(
                Finding(
                    "FAIL",
                    f"{label}.prompts.{name}",
                    f"stale: run used {sorted(ids)}, active is {active}",
                )
            )
    return out or [Finding("ok", f"{label}.prompts", "run matches the active prompt versions")]


def generation_findings(
    rows: list[dict], floors: dict, label: str, baseline: dict | None = None, tol: float = 0.05
) -> list[Finding]:
    rates = check_rates(rows)
    out = prompt_freshness(rows, label) + check_floors(label, rates, floors)
    if baseline:
        out += check_regression(label, rates, baseline, tol)
    return out


def judge_findings(rows: list[dict], floors: dict, label: str) -> list[Finding]:
    rates: dict[str, float | None] = {}
    for crit in floors:
        sel = [r for r in rows if r["criterion"] == crit and r.get("verdict")]
        rates[crit] = sum(r["verdict"] == "pass" for r in sel) / len(sel) if sel else None
    return check_floors(label, rates, floors)


def render(findings: list[Finding]) -> str:
    return "\n".join(f"{f.level:5} {f.where:42} {f.message}" for f in findings)


def exit_code(findings: list[Finding], strict: bool = False) -> int:
    """strict: skipped (uncalibrated) checks also fail, so a gate cannot pass by checking nothing."""
    bad = {"FAIL", "WARN"} if strict else {"FAIL"}
    return 1 if any(f.level in bad for f in findings) else 0


# ----------------------------- CLI -----------------------------


def _retrieval_cmd(args) -> list[Finding]:
    from scoper.evals.retrieval_eval import evaluate_retrieval
    from scoper.evals.runner import _fake_retriever, _runtime

    th = load_thresholds()["retrieval_fake" if args.fake else "retrieval_real"]
    retriever = _fake_retriever() if args.fake else _runtime()[2]
    out = []
    for label, path in (("main", CASES_PATH), ("hard", HARD_PATH)):
        report = evaluate_retrieval(retriever, load_cases(path))
        out += retrieval_findings(report, th[label], f"retrieval.{label}")
    return out


def _generation_cmd(args) -> list[Finding]:
    th = load_thresholds()
    rows = read_jsonl(Path(args.run) if args.run else RUNS_DIR / f"{args.system}_all.jsonl")
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        return [Finding("FAIL", "generation", f"no rows for {args.system}/{args.split}")]
    base_path = BASELINES_DIR / f"{args.system}_{args.split}.json"
    baseline = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else None
    if args.save_baseline:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(check_rates(rows), indent=1), encoding="utf-8")
        print(f"saved baseline {base_path}")
    return generation_findings(
        rows,
        th["generation"][args.system],
        f"generation.{args.system}.{args.split}",
        baseline,
        th["max_regression"],
    )


def _judge_cmd(args) -> list[Finding]:
    th = load_thresholds()["judge"]
    rows = [r for r in read_jsonl(RUNS_DIR / "judge_pointwise.jsonl") if r["system"] == "grounded"]
    return judge_findings(rows, th["grounded"], "judge.grounded")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--fake", action="store_true")
    r.set_defaults(fn=_retrieval_cmd)
    g = sub.add_parser("generation")
    g.add_argument("--system", choices=["grounded", "baseline"], default="grounded")
    g.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    g.add_argument("--save-baseline", action="store_true")
    g.add_argument(
        "--run", default=None, help="run file to gate (default: runs/<system>_all.jsonl)"
    )
    g.set_defaults(fn=_generation_cmd)
    sub.add_parser("judge").set_defaults(fn=_judge_cmd)
    ap.add_argument("--strict", action="store_true", help="treat skipped checks as failures")
    args = ap.parse_args(argv)
    findings = args.fn(args)
    print(render(findings))
    code = exit_code(findings, args.strict)
    skipped = sum(f.level == "WARN" for f in findings)
    print(
        "GATE:", "FAILED" if code else "passed", f"({skipped} check(s) skipped)" if skipped else ""
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
