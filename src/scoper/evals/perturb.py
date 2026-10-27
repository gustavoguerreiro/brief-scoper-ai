"""Planted-defect test for the judge: inject a KNOWN defect into a good output and see whether the
judge catches it. Ground truth is true by construction, so it does not depend on human labels
containing failures. Limits: the defects are crude and probably easier than natural errors, so
catching them proves sensitivity to obvious defects, not to subtle ones (e.g. implied causality).
"""

import copy
import math
import re

GENERIC_QUESTIONS = [
    "What is the budget?",
    "What is the deadline?",
    "Who are the stakeholders?",
    "What platforms must be supported?",
    "Is there an existing system?",
]
GENERIC_RISKS = ["Scope creep", "Communication gaps", "Unclear requirements", "Technical debt"]


def _fmt_variants(n: float) -> list[str]:
    i = round(n)
    return [f"{i:,}", str(i)]


def _candidates(sources: list[dict]) -> list[tuple[str, float]]:
    """Numbers a grounded output may quote about a past project: hours and cost, in two formats."""
    out = []
    for s in sources:
        for key in ("actual_hours", "actual_cost"):
            v = s.get(key)
            if v:
                out += [(t, v) for t in _fmt_variants(v)]
    return sorted(out, key=lambda t: -len(t[0]))  # longest first: "58,000" before "58"


def inject_number_swap(output: dict, sources: list[dict]) -> tuple[dict, bool]:
    """Replace a quoted source number in the summary or an assumption by a clearly wrong one."""
    out = copy.deepcopy(output)
    fields = [("summary", None)] + [
        ("assumption", i) for i in range(len((out.get("estimate") or {}).get("assumptions", [])))
    ]
    for token, value in _candidates(sources):
        pat = re.compile(rf"(?<![\d,.]){re.escape(token)}(?![\d,])")
        wrong = round(value * 1.7)
        repl = f"{wrong:,}" if "," in token else str(wrong)
        for kind, idx in fields:
            text = out["summary"] if kind == "summary" else out["estimate"]["assumptions"][idx]
            if pat.search(text):
                new = pat.sub(repl, text, count=1)
                if kind == "summary":
                    out["summary"] = new
                else:
                    out["estimate"]["assumptions"][idx] = new
                return out, True
    return out, False


def inject_invented_claim(output: dict, sources: list[dict]) -> dict:
    out = copy.deepcopy(output)
    title = (sources[0].get("title") if sources else None) or "closest past"
    out["summary"] = out["summary"].rstrip() + (
        f" The {title} project also won a regional industry award for its onboarding flow."
    )
    return out


def inject_boilerplate(output: dict) -> dict:
    out = copy.deepcopy(output)
    out["open_questions"] = list(GENERIC_QUESTIONS)
    out["risks"] = [
        {"description": d, "severity": "medium", "citations": []} for d in GENERIC_RISKS
    ]
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def summarize_planted(rows: list[dict], clean: dict[tuple[str, str], str]) -> dict:
    """rows: planted-judge results. clean: {(case_id, criterion): verdict} on the unmodified outputs."""
    out: dict = {}
    for kind in sorted({r["defect"] for r in rows}):
        sel = [r for r in rows if r["defect"] == kind and r["verdict"]]
        caught = sum(r["verdict"] == "fail" for r in sel)
        out[kind] = {
            "n": len(sel),
            "caught_as_fail": caught,
            "wilson95": list(wilson(caught, len(sel))),
        }
    for crit in ("grounding", "actionability"):
        ids = {r["case_id"] for r in rows if r["criterion"] == crit}
        v = [clean[(c, crit)] for c in ids if (c, crit) in clean]
        out[f"clean_{crit}"] = {
            "n": len(v),
            "passed": v.count("pass"),
            "wilson95": list(wilson(v.count("pass"), len(v))),
        }
    return out
