"""Import briefs from a folder of .md files.

uv run python -m scoper.ingestion.importer data/briefs --dry-run   # validate only, no DB/key
uv run python -m scoper.ingestion.importer data/briefs             # embed + store in Postgres
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from scoper.ingestion.chunker import chunk_brief, embed_text
from scoper.ingestion.models import BriefDoc, BriefParseError, parse_brief
from scoper.llm.embeddings import Embedder


@dataclass
class ImportReport:
    valid: list[BriefDoc] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # file name -> problem
    warnings: dict[str, str] = field(default_factory=dict)
    chunks: int = 0


def load_dir(path: Path) -> ImportReport:
    report, seen = ImportReport(), {}
    for f in sorted(path.glob("*.md")):
        if f.name.startswith("_"):  # _TEMPLATE.md and other private files
            continue
        try:
            doc = parse_brief(f.read_text(encoding="utf-8"))
        except BriefParseError as e:
            report.errors[f.name] = str(e)
            continue
        if doc.id in seen:
            report.errors[f.name] = f"duplicate id {doc.id!r} (also in {seen[doc.id]})"
            continue
        seen[doc.id] = f.name
        if doc.outcome == "delivered" and (doc.actual_hours is None or doc.actual_cost is None):
            report.warnings[f.name] = (
                "delivered but actual_hours/actual_cost missing: cannot ground estimates"
            )
        report.valid.append(doc)
    return report


def import_briefs(report: ImportReport, store, embedder: Embedder) -> None:
    for doc in report.valid:
        chunks = chunk_brief(doc)
        vectors = embedder.embed([embed_text(c) for c in chunks], run_id=f"import-{doc.id}")
        for c, v in zip(chunks, vectors):
            c.embedding = v
        store.add_brief(doc.meta(), doc.raw_text, chunks, embedder.model)
        report.chunks += len(chunks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="validate and count chunks only")
    args = ap.parse_args(argv)
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)

    report = load_dir(args.folder)
    for name, msg in report.errors.items():
        print(f"ERROR   {name}: {msg}")
    for name, msg in report.warnings.items():
        print(f"WARNING {name}: {msg}")
    print(f"{len(report.valid)} valid, {len(report.errors)} with errors")
    if report.errors:
        return 1
    if args.dry_run:
        n = sum(len(chunk_brief(d)) for d in report.valid)
        print(f"dry run: {n} chunks would be embedded; nothing written")
        return 0

    from scoper.llm.client import CallLog
    from scoper.llm.embeddings import OpenAIEmbedder
    from scoper.retrieval.pg_store import PgStore, database_url

    log = CallLog(path=Path("logs/llm_calls.jsonl"))
    embedder = OpenAIEmbedder(log)
    store = PgStore(database_url(), embedder.dim)
    store.init_schema()
    import_briefs(report, store, embedder)
    cost = sum(r.cost_usd for r in log.records)
    print(
        f"imported {len(report.valid)} briefs, {report.chunks} chunks, embedding cost ${cost:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
