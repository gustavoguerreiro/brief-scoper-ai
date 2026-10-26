"""Pure metric functions: retrieval ranking metrics and rater-agreement statistics."""

import math
import random
from collections.abc import Callable, Sequence

from scoper.retrieval.hybrid import Hit


def brief_ranking(hits: Sequence[Hit]) -> list[str]:
    """Chunk hits -> brief ranking, in order of each brief's best-ranked chunk."""
    seen: list[str] = []
    for h in hits:
        if h.chunk.brief_id not in seen:
            seen.append(h.chunk.brief_id)
    return seen


def hit_at_k(ranking: Sequence[str], gold: Sequence[str], k: int) -> float:
    return 1.0 if any(b in gold for b in ranking[:k]) else 0.0


def reciprocal_rank(ranking: Sequence[str], gold: Sequence[str]) -> float:
    for i, b in enumerate(ranking, start=1):
        if b in gold:
            return 1.0 / i
    return 0.0


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """kappa = (po - pe) / (1 - pe). NaN when both raters use one single label (pe == 1)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    if math.isclose(pe, 1.0):
        return float("nan")
    return (po - pe) / (1 - pe)


def bootstrap_ci(
    a: Sequence[str],
    b: Sequence[str],
    stat: Callable[[Sequence[str], Sequence[str]], float] = cohen_kappa,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap over label pairs. With ~20 pairs the interval is wide: report it."""
    rng = random.Random(seed)
    idx = range(len(a))
    vals = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        v = stat([a[i] for i in s], [b[i] for i in s])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return vals[int(alpha / 2 * len(vals))], vals[
        min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))
    ]


def confusion(a: Sequence[str], b: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x, y in zip(a, b, strict=True):
        key = f"human={x}|judge={y}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))
