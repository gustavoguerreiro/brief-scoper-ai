import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / "evals" / "cases.jsonl"
HARD_PATH = ROOT / "evals" / "retrieval_hard.jsonl"
RUNS_DIR = ROOT / "evals" / "runs"
LABELS_DIR = ROOT / "evals" / "labels"


class EvalCase(BaseModel):
    id: str
    brief: str
    tags: list[str]
    gold_brief_ids: list[str]  # past briefs a person would call relevant ([] = nothing relevant)
    expect_abstain: bool  # the right answer is "not enough information" (no estimate)
    canary: str | None  # string that must NOT appear in the output (prompt-injection probe)
    split: Literal["dev", "test"]  # tune on dev, report on test


def load_cases(path: Path = CASES_PATH, split: str = "all") -> list[EvalCase]:
    cases = [
        EvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate eval case ids")
    return cases if split == "all" else [c for c in cases if c.split == split]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
