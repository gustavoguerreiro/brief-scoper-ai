"""Interactive hand-labelling of the sample, blind to the judge.

    uv run python -m scoper.evals.label

For each sampled output you answer, per criterion: p (pass), f (fail), s (skip for now), q (quit).
Progress is saved after every answer, so you can stop and resume.
"""

import json
import sys

from scoper.evals.dataset import LABELS_DIR, RUNS_DIR, load_cases, read_jsonl
from scoper.evals.judge import RUBRICS, render_sources_text

CRITERIA = ("grounding", "actionability")


def _show(case, row) -> None:
    bar = "=" * 78
    print(f"\n{bar}\nCASE {case.id}  tags={case.tags}\n{bar}")
    print("NEW BRIEF:\n" + case.brief)
    print("\n" + render_sources_text(row["sources"]))
    print("\nOUTPUT:\n" + json.dumps(row["output"], indent=1, ensure_ascii=False))


def main() -> int:
    sample = json.loads((LABELS_DIR / "sample.json").read_text(encoding="utf-8"))
    cases = {c.id: c for c in load_cases()}
    rows = {r["case_id"]: r for r in read_jsonl(RUNS_DIR / "grounded_all.jsonl")}
    path = LABELS_DIR / "human.jsonl"
    done = {(r["case_id"], r["criterion"]) for r in read_jsonl(path)}
    todo = [(cid, crit) for cid in sample for crit in CRITERIA if (cid, crit) not in done]
    print(f"{len(done)} labels saved, {len(todo)} to go. Rubrics:")
    for c in CRITERIA:
        print(f"\n[{c}] {RUBRICS[c]}")
    shown = None
    for cid, crit in todo:
        if shown != cid:
            _show(cases[cid], rows[cid])
            shown = cid
        while True:
            try:
                ans = input(f"\n{cid} / {crit}  -> p / f / s / q: ").strip().lower()
            except EOFError:
                return 0
            if ans in ("p", "f", "s", "q"):
                break
        if ans == "q":
            return 0
        if ans == "s":
            continue
        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"case_id": cid, "criterion": crit, "label": "pass" if ans == "p" else "fail"}
                )
                + "\n"
            )
    print("All labels done. Next: uv run python -m scoper.evals.runner calibrate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
