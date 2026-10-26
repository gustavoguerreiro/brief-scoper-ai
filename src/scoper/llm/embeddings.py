"""Embedding seam, mirroring the ModelClient idea: one protocol, a fake for tests, one provider."""

import hashlib
import math
import re
import time
from typing import Protocol

from scoper.llm.client import CallLog, CallRecord, compute_cost, load_prices

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str], run_id: str = "") -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic hashed bag-of-words, L2-normalised. Similar wording => similar vectors.

    It captures lexical overlap only, never synonyms, so tests built on it can't claim
    anything about real semantic retrieval quality.
    """

    model = "fake-embedding"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str], run_id: str = "") -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in _WORD.findall(t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class OpenAIEmbedder:
    model = "text-embedding-3-small"
    dim = 1536

    def __init__(self, log: CallLog, api_key: str | None = None, prices: dict | None = None):
        import openai  # local import: offline tests never need the SDK

        self._client = openai.OpenAI(api_key=api_key, timeout=60.0)  # SDK default retries (2)
        self._log = log
        self._prices = prices if prices is not None else load_prices()

    def embed(self, texts: list[str], run_id: str = "") -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 96):
            batch = texts[i : i + 96]
            start = time.perf_counter()
            try:
                resp = self._client.embeddings.create(model=self.model, input=batch)
            except Exception as e:
                self._log.add(
                    CallRecord(run_id, "embed", self.model, 0, 0, 0.0, 0.0, "error", repr(e)[:300])
                )
                raise
            tokens = resp.usage.total_tokens
            self._log.add(
                CallRecord(
                    run_id,
                    "embed",
                    self.model,
                    tokens,
                    0,
                    compute_cost(self._prices, self.model, tokens, 0),
                    (time.perf_counter() - start) * 1000,
                    "ok",
                )
            )
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return vectors
