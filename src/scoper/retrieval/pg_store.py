"""PostgreSQL + pgvector store. Same interface as InMemoryStore."""

import os
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from scoper.retrieval.store import Chunk, Filters

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "db" / "schema.sql"

DEFAULT_DSN = "postgresql://scoper:scoper@localhost:5440/scoper"  # matches docker-compose.yml


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DSN)


_COLS = (
    "c.id, c.brief_id, c.section, c.ordinal, c.text, b.title, b.industry, b.project_type, "
    "b.budget_band, b.outcome, b.actual_hours, b.actual_cost, b.visibility"
)
_FROM = "FROM chunks c JOIN briefs b ON b.id = c.brief_id"


def _chunk(row) -> Chunk:
    meta = {
        "id": row[1],
        "title": row[5],
        "industry": row[6],
        "project_type": row[7],
        "budget_band": row[8],
        "outcome": row[9],
        "actual_hours": row[10],
        "actual_cost": row[11],
        "visibility": row[12],
    }
    return Chunk(row[0], row[1], row[2], row[3], row[4], meta)


class PgStore:
    def __init__(self, dsn: str, dim: int):
        self.dsn, self.dim = dsn, dim

    def _conn(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn

    def init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8").replace("{dim}", str(self.dim))
        with psycopg.connect(self.dsn) as conn:  # extension must exist before register_vector
            conn.execute(sql)

    def reset(self) -> None:
        with self._conn() as conn:
            conn.execute("TRUNCATE chunks, briefs CASCADE")

    def add_brief(self, meta: dict, raw_text: str, chunks: list[Chunk], embed_model: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO briefs (id, title, raw_text, industry, project_type, budget_band,
                       outcome, actual_hours, actual_cost, visibility)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, raw_text=EXCLUDED.raw_text,
                       industry=EXCLUDED.industry, project_type=EXCLUDED.project_type,
                       budget_band=EXCLUDED.budget_band, outcome=EXCLUDED.outcome,
                       actual_hours=EXCLUDED.actual_hours, actual_cost=EXCLUDED.actual_cost,
                       visibility=EXCLUDED.visibility""",
                (
                    meta["id"],
                    meta["title"],
                    raw_text,
                    meta["industry"],
                    meta["project_type"],
                    meta["budget_band"],
                    meta["outcome"],
                    meta["actual_hours"],
                    meta["actual_cost"],
                    meta["visibility"],
                ),
            )
            conn.execute("DELETE FROM chunks WHERE brief_id = %s", (meta["id"],))
            for c in chunks:
                conn.execute(
                    """INSERT INTO chunks (id, brief_id, section, ordinal, text, embed_model, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        c.id,
                        c.brief_id,
                        c.section,
                        c.ordinal,
                        c.text,
                        embed_model,
                        np.array(c.embedding, dtype=np.float32),
                    ),
                )

    def list_chunks(self, filters: Filters) -> list[Chunk]:
        where, params = filters.to_sql()
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {_COLS} {_FROM} WHERE {where} ORDER BY c.brief_id, c.section, c.ordinal",
                params,
            ).fetchall()
        return [_chunk(r) for r in rows]

    def vector_search(
        self, qvec: list[float], filters: Filters, k: int
    ) -> list[tuple[Chunk, float]]:
        where, params = filters.to_sql()
        q = np.array(qvec, dtype=np.float32)
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT {_COLS}, 1 - (c.embedding <=> %s) AS sim {_FROM}
                    WHERE {where} ORDER BY c.embedding <=> %s LIMIT %s""",
                [q, *params, q, k],
            ).fetchall()
        return [(_chunk(r), float(r[13])) for r in rows]

    def count(self) -> tuple[int, int]:
        with self._conn() as conn:
            b = conn.execute("SELECT count(*) FROM briefs").fetchone()[0]
            c = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return b, c
