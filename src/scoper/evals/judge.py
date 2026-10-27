"""LLM-as-judge. One criterion per call, explicit rubric, reason before verdict.

Two modes:
- pointwise: grade ONE output pass/fail against a rubric.
- pairwise: pick the better of two outputs; always run in both orders to expose position bias.

Known judge biases to keep in mind: position (prefers the first/last answer), verbosity (prefers
longer), self-preference (prefers its own family's style), and leniency. The judge model is
therefore a different tier than the generator, and its verdicts are only trusted after
calibration against human labels (see calibration_report in runner.py).
"""

import json
from typing import Literal

from pydantic import BaseModel

from scoper.generation.structured import parse_model, structured_call
from scoper.llm.client import CallLog, ModelClient
from scoper.prompts.registry import get_prompt

CRITERIA = ("grounding", "actionability")


def get_rubric(criterion: str) -> str:
    return get_prompt(f"rubric_{criterion}").text


def _system(kind: str, criterion: str) -> str:
    return get_prompt(f"judge_{kind}").render(criterion=criterion, rubric=get_rubric(criterion))


class PointVerdict(BaseModel):
    reason: str
    verdict: Literal["pass", "fail"]


class PairVerdict(BaseModel):
    reason: str
    winner: Literal["A", "B", "tie"]


def render_sources_text(sources: list[dict]) -> str:
    if not sources:
        return "<sources>\n(none)\n</sources>"
    parts = [
        f'[{s["id"]}] project="{s.get("title", "")}" outcome={s.get("outcome", "")} '
        f"actual_hours={s.get('actual_hours')} actual_cost={s.get('actual_cost')}\n{s['text']}"
        for s in sources
    ]
    return "<sources>\n" + "\n\n".join(parts) + "\n</sources>"


def _dump(output: dict) -> str:
    return json.dumps(output, ensure_ascii=False, indent=1)


def judge_pointwise(
    client: ModelClient,
    log: CallLog,
    criterion: str,
    brief: str,
    sources: list[dict],
    output: dict,
    run_id: str,
    model: str,
) -> PointVerdict:
    user = (
        f"<brief>\n{brief}\n</brief>\n\n{render_sources_text(sources)}\n\n"
        f"<output>\n{_dump(output)}\n</output>"
    )
    return structured_call(
        client,
        log,
        system=_system("pointwise", criterion),
        messages=[{"role": "user", "content": user}],
        parse=parse_model(PointVerdict),
        run_id=run_id,
        model=model,
        max_repairs=1,
        cost_cap_usd=0.5,
        max_tokens=1500,
        effort="low",
    )


def judge_pair_once(
    client: ModelClient,
    log: CallLog,
    criterion: str,
    brief: str,
    a: dict,
    b: dict,
    run_id: str,
    model: str,
) -> PairVerdict:
    user = (
        f'<brief>\n{brief}\n</brief>\n\n<output id="A">\n{_dump(a)}\n</output>\n\n'
        f'<output id="B">\n{_dump(b)}\n</output>'
    )
    return structured_call(
        client,
        log,
        system=_system("pairwise", criterion),
        messages=[{"role": "user", "content": user}],
        parse=parse_model(PairVerdict),
        run_id=run_id,
        model=model,
        max_repairs=1,
        cost_cap_usd=0.5,
        max_tokens=1500,
        effort="low",
    )


def judge_pair_both_orders(
    client: ModelClient,
    log: CallLog,
    criterion: str,
    brief: str,
    x: dict,
    y: dict,
    run_id: str,
    model: str,
) -> dict:
    """x and y are two systems' outputs. Run (A=x, B=y) then (A=y, B=x)."""
    v1 = judge_pair_once(client, log, criterion, brief, x, y, run_id + "-xy", model)
    v2 = judge_pair_once(client, log, criterion, brief, y, x, run_id + "-yx", model)
    to_system = {
        "xy": {"A": "x", "B": "y", "tie": "tie"},
        "yx": {"A": "y", "B": "x", "tie": "tie"},
    }
    w1, w2 = to_system["xy"][v1.winner], to_system["yx"][v2.winner]
    return {
        "raw_first_order": v1.winner,
        "raw_swapped_order": v2.winner,
        "system_first_order": w1,
        "system_swapped_order": w2,
        "consistent": w1 == w2,
        "reason_first": v1.reason,
        "reason_swapped": v2.reason,
    }


def position_bias_report(pair_rows: list[dict]) -> dict:
    """If the judge ignored position, swapping the order would swap the raw letter it picks.

    same_position_both_orders: it picked the same LETTER both times (e.g. A then A), which means
    the verdict followed the slot, not the content.
    """
    n = len(pair_rows)
    raw = [r[k] for r in pair_rows for k in ("raw_first_order", "raw_swapped_order")]
    same_slot = sum(
        r["raw_first_order"] == r["raw_swapped_order"] and r["raw_first_order"] != "tie"
        for r in pair_rows
    )
    return {
        "n_cases": n,
        "consistent_rate": sum(r["consistent"] for r in pair_rows) / n if n else None,
        "same_position_both_orders": same_slot,
        "picked_A_share": raw.count("A") / len(raw) if raw else None,
        "picked_B_share": raw.count("B") / len(raw) if raw else None,
        "tie_share": raw.count("tie") / len(raw) if raw else None,
    }
