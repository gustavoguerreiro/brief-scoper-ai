"""Needs the compose Postgres: docker compose up -d db && uv run pytest -m db"""

import os

import psycopg
import pytest

from scoper.ingestion.chunker import chunk_brief, embed_text
from scoper.ingestion.models import parse_brief
from scoper.llm.embeddings import FakeEmbedder
from scoper.retrieval.hybrid import Retriever
from scoper.retrieval.pg_store import PgStore
from scoper.retrieval.store import Filters
from tests.test_m2_retrieval import CORPUS, brief_text

pytestmark = pytest.mark.db
DSN = os.getenv("TEST_DATABASE_URL", "postgresql://scoper:scoper@localhost:5440/scoper")


@pytest.fixture
def store():
    try:
        psycopg.connect(DSN, connect_timeout=3).close()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable (docker compose up -d db)")
    emb = FakeEmbedder()
    # separate table dimension from the real 1536 schema: use a throwaway schema
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DROP SCHEMA IF EXISTS t CASCADE; CREATE SCHEMA t")
    s = PgStore(DSN + ("&" if "?" in DSN else "?") + "options=-csearch_path%3Dt,public", emb.dim)
    s.init_schema()
    for text in CORPUS + (brief_text("priv", visibility="private"),):
        doc = parse_brief(text)
        chunks = chunk_brief(doc)
        for ch, v in zip(chunks, emb.embed([embed_text(x) for x in chunks]), strict=True):
            ch.embedding = v
        s.add_brief(doc.meta(), doc.raw_text, chunks, emb.model)
    yield s, emb
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DROP SCHEMA t CASCADE")


def test_counts_and_roundtrip(store):
    s, _ = store
    assert s.count() == (4, 16)
    ch = s.list_chunks(Filters(visibility=("team", "private")))[0]
    assert ch.meta["title"] and ch.text


def test_private_hidden_by_default(store):
    s, _ = store
    assert "priv" not in {c.brief_id for c in s.list_chunks(Filters())}
    assert "priv" in {c.brief_id for c in s.list_chunks(Filters(visibility=("team", "private")))}


def test_hybrid_search_in_postgres_matches_expectations(store):
    s, emb = store
    hits = Retriever(s, emb).search("GPS tracking of trucks dashboard", k=3)
    assert hits[0].chunk.brief_id == "logi" and set(hits[0].ranks) == {"bm25", "vector"}


def test_filters_in_sql_and_leave_one_out(store):
    s, emb = store
    r = Retriever(s, emb)
    assert all(
        h.chunk.meta["industry"] == "retail"
        for h in r.search("booking", Filters(industry=("retail",)))
    )
    assert all(
        h.chunk.brief_id != "shop"
        for h in r.search("payment cart", Filters(exclude_brief_ids=("shop",)))
    )


def test_reimport_replaces_chunks(store):
    s, emb = store
    doc = parse_brief(brief_text("health", scope="Totally different scope text."))
    chunks = chunk_brief(doc)
    for ch, v in zip(chunks, emb.embed([embed_text(x) for x in chunks]), strict=True):
        ch.embedding = v
    s.add_brief(doc.meta(), doc.raw_text, chunks, emb.model)
    assert s.count() == (4, 16)
