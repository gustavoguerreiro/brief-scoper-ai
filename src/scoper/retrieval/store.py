"""Chunk model, metadata filters (including the permission-style one) and an in-memory store."""

import math
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str  # "<brief_id>:<section>:<n>", readable so citations are checkable by eye
    brief_id: str
    section: str
    ordinal: int
    text: str
    meta: dict = field(default_factory=dict)  # title, industry, project_type, ..., actual_*
    embedding: list[float] | None = None


@dataclass(frozen=True)
class Filters:
    industry: tuple[str, ...] = ()
    project_type: tuple[str, ...] = ()
    budget_band: tuple[str, ...] = ()
    outcome: tuple[str, ...] = ()
    # Permission-style filter: the caller's visible scopes. Private briefs are invisible by
    # default. In a real system this comes from authentication, never from the request body.
    visibility: tuple[str, ...] = ("team",)
    # Leave-one-out for evals: a brief must not retrieve itself.
    exclude_brief_ids: tuple[str, ...] = ()

    def matches(self, meta: dict, brief_id: str) -> bool:
        if meta.get("visibility") not in self.visibility:
            return False
        if brief_id in self.exclude_brief_ids:
            return False
        for name in ("industry", "project_type", "budget_band", "outcome"):
            wanted = getattr(self, name)
            if wanted and meta.get(name) not in wanted:
                return False
        return True

    def to_sql(self) -> tuple[str, list]:
        """WHERE clause over alias `b` (briefs). Applied before ranking, in the database."""
        if not self.visibility:
            return "FALSE", []
        conds, params = ["b.visibility = ANY(%s)"], [list(self.visibility)]
        for name in ("industry", "project_type", "budget_band", "outcome"):
            wanted = getattr(self, name)
            if wanted:
                conds.append(f"b.{name} = ANY(%s)")
                params.append(list(wanted))
        if self.exclude_brief_ids:
            conds.append("NOT (b.id = ANY(%s))")
            params.append(list(self.exclude_brief_ids))
        return " AND ".join(conds), params


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class InMemoryStore:
    """Same interface as PgStore; used by offline tests and eval experiments."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._briefs: dict[str, dict] = {}

    def add_brief(self, meta: dict, raw_text: str, chunks: list[Chunk], embed_model: str) -> None:
        self._briefs[meta["id"]] = meta
        self._chunks = [c for c in self._chunks if c.brief_id != meta["id"]] + chunks

    def list_chunks(self, filters: Filters) -> list[Chunk]:
        return [c for c in self._chunks if filters.matches(c.meta, c.brief_id)]

    def vector_search(
        self, qvec: list[float], filters: Filters, k: int
    ) -> list[tuple[Chunk, float]]:
        scored = [(c, cosine(qvec, c.embedding)) for c in self.list_chunks(filters) if c.embedding]
        return sorted(scored, key=lambda t: t[1], reverse=True)[:k]

    def count(self) -> tuple[int, int]:
        return len(self._briefs), len(self._chunks)
