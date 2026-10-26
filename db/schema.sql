-- {dim} is substituted by PgStore.init_schema() from the embedder's dimension.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS briefs (
    id            text PRIMARY KEY,
    title         text NOT NULL,
    raw_text      text NOT NULL,
    industry      text NOT NULL,
    project_type  text NOT NULL,
    budget_band   text NOT NULL,
    outcome       text NOT NULL,
    actual_hours  double precision,
    actual_cost   double precision,
    visibility    text NOT NULL DEFAULT 'team',
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          text PRIMARY KEY,
    brief_id    text NOT NULL REFERENCES briefs(id) ON DELETE CASCADE,
    section     text NOT NULL,
    ordinal     int  NOT NULL,
    text        text NOT NULL,
    embed_model text NOT NULL,
    embedding   vector({dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_brief_idx ON chunks (brief_id);
-- Unneeded at ~150 chunks (an exact scan is fast); kept to practise the production setup.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
