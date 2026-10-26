"""Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion."""

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from rank_bm25 import BM25Okapi

from scoper.llm.embeddings import Embedder
from scoper.retrieval.store import Chunk, Filters

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "with",
        "our",
        "you",
        "your",
        "i",
        "they",
        "their",
        "them",
        "will",
        "would",
        "can",
        "could",
        "should",
    ]
)

Mode = Literal["hybrid", "bm25", "vector"]


class Store(Protocol):
    def list_chunks(self, filters: Filters) -> list[Chunk]: ...
    def vector_search(
        self, qvec: list[float], filters: Filters, k: int
    ) -> list[tuple[Chunk, float]]: ...


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # {"bm25": 2, "vector": 1}


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, rrf_k: int = 60, candidate_k: int = 20):
        self.store, self.embedder = store, embedder
        self.rrf_k, self.candidate_k = rrf_k, candidate_k

    def _bm25(self, query: str, filters: Filters) -> list[tuple[Chunk, float]]:
        chunks = self.store.list_chunks(filters)
        q = tokenize(query)
        if not chunks or not q:
            return []
        # Rebuilt per query: fine for ~100s of chunks; would need caching/invalidation at scale.
        bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        scored = [(c, float(s)) for c, s in zip(chunks, bm25.get_scores(q)) if s > 0]
        return sorted(scored, key=lambda t: t[1], reverse=True)[: self.candidate_k]

    def _vector(self, query: str, filters: Filters, run_id: str) -> list[tuple[Chunk, float]]:
        qvec = self.embedder.embed([query], run_id=run_id)[0]
        return self.store.vector_search(qvec, filters, self.candidate_k)

    def search(
        self,
        query: str,
        filters: Filters | None = None,
        k: int = 5,
        mode: Mode = "hybrid",
        run_id: str = "",
        per_brief_cap: int | None = None,
    ) -> list[Hit]:
        filters = filters or Filters()
        lists: dict[str, list[tuple[Chunk, float]]] = {}
        if mode in ("hybrid", "bm25"):
            lists["bm25"] = self._bm25(query, filters)
        if mode in ("hybrid", "vector"):
            lists["vector"] = self._vector(query, filters, run_id)

        by_id = {c.id: c for items in lists.values() for c, _ in items}
        ranks = {
            name: {c.id: r for r, (c, _) in enumerate(items, start=1)}
            for name, items in lists.items()
        }
        if mode == "hybrid":
            fused = rrf([[c.id for c, _ in items] for items in lists.values()], self.rrf_k)
        else:
            (only,) = lists.values()
            fused = {c.id: s for c, s in only}
        order = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        if per_brief_cap:  # diversity: at most N chunks from the same brief
            seen: dict[str, int] = {}
            kept = []
            for cid in order:
                b = by_id[cid].brief_id
                if seen.get(b, 0) < per_brief_cap:
                    kept.append(cid)
                    seen[b] = seen.get(b, 0) + 1
            order = kept
        order = order[:k]
        return [
            Hit(by_id[cid], fused[cid], {n: r[cid] for n, r in ranks.items() if cid in r})
            for cid in order
        ]
