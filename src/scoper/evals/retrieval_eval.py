"""Retrieval evaluation: no LLM involved, so it is cheap, deterministic and runs offline."""

from scoper.evals.dataset import EvalCase
from scoper.evals.metrics import brief_ranking, hit_at_k, reciprocal_rank
from scoper.retrieval.hybrid import Retriever

CONFIGS: dict[str, dict] = {
    "bm25": {"mode": "bm25"},
    "vector": {"mode": "vector"},
    "hybrid": {"mode": "hybrid"},
    "hybrid_cap2": {"mode": "hybrid", "per_brief_cap": 2},  # at most 2 chunks per brief
}


class CachingEmbedder:
    """Embeds each distinct query once, so comparing configs doesn't repeat paid calls."""

    def __init__(self, inner):
        self._inner, self._cache = inner, {}
        self.model, self.dim = inner.model, inner.dim

    def embed(self, texts: list[str], run_id: str = "") -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self._inner.embed(missing, run_id=run_id), strict=True):
                self._cache[t] = v
        return [self._cache[t] for t in texts]


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[EvalCase],
    configs: dict[str, dict] | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
    top_chunks: int = 5,
) -> dict:
    """Only cases with at least one gold brief are scored (vague/out-of-domain have none)."""
    scored = [c for c in cases if c.gold_brief_ids]
    report: dict = {"n_cases": len(scored), "configs": {}}
    for name, cfg in (configs or CONFIGS).items():
        per_case, hits_k, rrs, distinct, gold_in_top = {}, {k: [] for k in ks}, [], [], []
        for c in scored:
            hits = retriever.search(c.brief, k=60, run_id=f"reteval-{c.id}", **cfg)
            ranking = brief_ranking(hits)
            for k in ks:
                hits_k[k].append(hit_at_k(ranking, c.gold_brief_ids, k))
            rrs.append(reciprocal_rank(ranking, c.gold_brief_ids))
            top = hits[:top_chunks]
            distinct.append(len({h.chunk.brief_id for h in top}))
            gold_in_top.append(
                1.0 if any(h.chunk.brief_id in c.gold_brief_ids for h in top) else 0.0
            )
            per_case[c.id] = {"top_briefs": ranking[:5], "gold": c.gold_brief_ids}
        n = len(scored)
        report["configs"][name] = {
            **{f"hit@{k}": sum(v) / n for k, v in hits_k.items()},
            "mrr": sum(rrs) / n,
            f"gold_in_top{top_chunks}_chunks": sum(gold_in_top) / n,
            f"distinct_briefs_in_top{top_chunks}_chunks": sum(distinct) / n,
            "per_case": per_case,
        }
    return report


def format_report(report: dict) -> str:
    cfgs = report["configs"]
    cols = [k for k in next(iter(cfgs.values())) if k != "per_case"]
    lines = [
        f"cases scored: {report['n_cases']}",
        "config".ljust(14) + "".join(c.rjust(26) for c in cols),
    ]
    for name, m in cfgs.items():
        lines.append(name.ljust(14) + "".join(f"{m[c]:.3f}".rjust(26) for c in cols))
    return "\n".join(lines)
