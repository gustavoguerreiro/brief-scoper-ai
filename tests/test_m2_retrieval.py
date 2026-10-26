from pathlib import Path

import pytest

from scoper.ingestion.chunker import chunk_brief, embed_text
from scoper.ingestion.importer import import_briefs, load_dir
from scoper.ingestion.models import BriefParseError, parse_brief
from scoper.llm.embeddings import FakeEmbedder
from scoper.retrieval.hybrid import Retriever, rrf, tokenize
from scoper.retrieval.store import Filters, InMemoryStore

ROOT = Path(__file__).resolve().parents[1]


def brief_text(
    id="b1",
    title="Clinic booking",
    industry="healthcare",
    ptype="web-app",
    visibility="team",
    outcome_text="Delivered in 600 hours.",
    scope="Online booking and SMS reminders.",
    actual_hours="600",
    actual_cost="55000",
):
    return f"""---
id: {id}
title: {title}
industry: {industry}
project_type: {ptype}
budget_band: 50k-100k
outcome: delivered
actual_hours: {actual_hours}
actual_cost: {actual_cost}
visibility: {visibility}
---

## Context
A clinic group needed a better way to manage appointments.

## Scope
{scope}

## Constraints
Launch before March.

## Outcome
{outcome_text}
"""


def build_store(*texts):
    store, emb = InMemoryStore(), FakeEmbedder()
    d = [parse_brief(t) for t in texts]
    from scoper.ingestion.importer import ImportReport

    rep = ImportReport(valid=d)
    import_briefs(rep, store, emb)
    return store, emb


# ---- parsing / import ----


def test_parse_valid_brief():
    d = parse_brief(brief_text())
    assert d.id == "b1" and d.actual_hours == 600 and set(d.sections) >= {"context", "outcome"}


@pytest.mark.parametrize(
    "bad,msg",
    [
        ("no front matter", "front matter"),
        (brief_text(ptype="spaceship"), "project_type"),
        (brief_text().replace("## Outcome\nDelivered in 600 hours.\n", ""), "outcome"),
    ],
)
def test_parse_errors_are_readable(bad, msg):
    with pytest.raises(BriefParseError, match=msg):
        parse_brief(bad)


def test_the_shipped_template_stays_valid(tmp_path):
    (tmp_path / "b001.md").write_text(
        (ROOT / "data/briefs/_TEMPLATE.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    rep = load_dir(tmp_path)
    assert not rep.errors and rep.valid[0].id == "b001"


def test_load_dir_skips_underscore_flags_duplicates_and_warns(tmp_path):
    (tmp_path / "_TEMPLATE.md").write_text("ignored", encoding="utf-8")
    (tmp_path / "a.md").write_text(brief_text("x"), encoding="utf-8")
    (tmp_path / "b.md").write_text(brief_text("x"), encoding="utf-8")
    (tmp_path / "c.md").write_text(
        brief_text("y", actual_hours="", actual_cost=""), encoding="utf-8"
    )
    rep = load_dir(tmp_path)
    assert [d.id for d in rep.valid] == ["x", "y"]
    assert "duplicate id" in rep.errors["b.md"]
    assert "c.md" in rep.warnings


# ---- chunking ----


def test_chunk_ids_sections_and_prefix():
    doc = parse_brief(brief_text())
    chunks = chunk_brief(doc)
    assert [c.id for c in chunks] == [
        "b1:context:0",
        "b1:scope:0",
        "b1:constraints:0",
        "b1:outcome:0",
    ]
    assert embed_text(chunks[1]).startswith("Clinic booking | scope")
    assert chunks[0].meta["industry"] == "healthcare"


def test_long_section_is_split_with_overlap():
    long = " ".join(f"w{i}" for i in range(450))
    doc = parse_brief(brief_text(outcome_text=long))
    outs = [c for c in chunk_brief(doc, max_words=200, overlap=30) if c.section == "outcome"]
    assert len(outs) == 3
    assert all(len(c.text.split()) <= 200 for c in outs)
    assert outs[0].text.split()[-30:] == outs[1].text.split()[:30]


def test_paragraphs_are_packed_not_split():
    paras = "\n\n".join(f"para{i} " + "x " * 60 for i in range(5))  # 61 words each
    doc = parse_brief(brief_text(outcome_text=paras))
    outs = [c for c in chunk_brief(doc, max_words=200) if c.section == "outcome"]
    assert [len(c.text.split("\n\n")) for c in outs] == [3, 2]


# ---- fusion / filters ----


def test_rrf_rewards_agreement():
    s = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert s["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["c"] == pytest.approx(1 / 63) and s["d"] == pytest.approx(1 / 63)
    assert s["a"] > s["c"]
    assert set(s) == {"a", "b", "c", "d"}


def test_tokenize_drops_stopwords():
    assert tokenize("The booking of an App!") == ["booking", "app"]


def test_filters_private_hidden_by_default_and_exclusion():
    store, _emb = build_store(brief_text("pub"), brief_text("priv", visibility="private"))
    assert {c.brief_id for c in store.list_chunks(Filters())} == {"pub"}
    assert {c.brief_id for c in store.list_chunks(Filters(visibility=("team", "private")))} == {
        "pub",
        "priv",
    }
    assert {
        c.brief_id
        for c in store.list_chunks(
            Filters(visibility=("team", "private"), exclude_brief_ids=("pub",))
        )
    } == {"priv"}
    assert store.list_chunks(Filters(visibility=())) == []


def test_filters_to_sql_shape():
    where, params = Filters(industry=("retail",), exclude_brief_ids=("b9",)).to_sql()
    assert where.count("%s") == len(params) == 3 and "b.industry" in where


# ---- retrieval ----

CORPUS = (
    brief_text(
        "health",
        "Clinic booking",
        "healthcare",
        scope="Online booking and SMS reminders for patients.",
    ),
    brief_text(
        "shop",
        "Shoe store",
        "retail",
        "ecommerce",
        scope="Product catalog, cart and payment gateway integration.",
    ),
    brief_text(
        "logi",
        "Fleet tracker",
        "logistics",
        "data-platform",
        scope="Real-time truck GPS tracking dashboards.",
    ),
)


def test_bm25_finds_keyword_match():
    store, emb = build_store(*CORPUS)
    hits = Retriever(store, emb).search("payment gateway cart", k=3, mode="bm25")
    assert (
        hits[0].chunk.brief_id == "shop"
        and "bm25" in hits[0].ranks
        and "vector" not in hits[0].ranks
    )


def test_vector_finds_similar_text():
    store, emb = build_store(*CORPUS)
    hits = Retriever(store, emb).search("GPS tracking of trucks on a dashboard", k=3, mode="vector")
    assert hits[0].chunk.brief_id == "logi"


def test_hybrid_combines_and_reports_ranks():
    store, emb = build_store(*CORPUS)
    hits = Retriever(store, emb).search("booking reminders patients sms", k=3)
    assert hits[0].chunk.brief_id == "health"
    assert set(hits[0].ranks) == {"bm25", "vector"}
    assert hits[0].score >= hits[-1].score


def test_filters_apply_before_ranking():
    store, emb = build_store(*CORPUS)
    hits = Retriever(store, emb).search("booking sms", Filters(industry=("retail",)), k=5)
    assert hits and all(h.chunk.meta["industry"] == "retail" for h in hits)


def test_leave_one_out_and_private_never_leak():
    store, emb = build_store(
        brief_text("a"),
        brief_text("p", visibility="private", scope="Online booking and SMS reminders."),
    )
    hits = Retriever(store, emb).search(
        "online booking sms reminders", Filters(exclude_brief_ids=("a",))
    )
    assert hits == []


def test_empty_store_and_empty_query():
    store, emb = build_store()
    r = Retriever(store, emb)
    assert r.search("anything") == []
    store, emb = build_store(*CORPUS)
    assert Retriever(store, emb).search("the of and", mode="bm25") == []
