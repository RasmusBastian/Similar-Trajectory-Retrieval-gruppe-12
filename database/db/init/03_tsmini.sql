-- TSMini model outputs. Shared dataset inputs and ground truth live in db/init/02_data.sql.
-- Filled by TSMini/import_to_db.py (docker compose run --rm --build tsmini-import).
-- Also run by that script, so it must stay re-runnable (IF NOT EXISTS / OR REPLACE).

CREATE SCHEMA IF NOT EXISTS tsmini;

-- Output: 128-dimensional embedding of every trajectory (exp/embeddings_<dataset>.npz).
-- The L1 distance between two embeddings approximates the measure the model was trained on.
CREATE TABLE IF NOT EXISTS tsmini.embedding (
    dataset   TEXT NOT NULL,
    model     TEXT NOT NULL,                     -- checkpoint file name in tsmini.file
    idx       INT NOT NULL,
    embedding REAL[] NOT NULL,
    PRIMARY KEY (dataset, model, idx),
    FOREIGN KEY (dataset, idx) REFERENCES data.trajectory ON DELETE CASCADE
);

-- Output: training logs (exp/log/*.log)
CREATE TABLE IF NOT EXISTS tsmini.training_log (
    file_name TEXT PRIMARY KEY,
    content   TEXT NOT NULL
);

-- Output: other files, e.g. the trained model (exp/snapshots/*.pt) and plots. path is relative to TSMini/.
-- Save one back to disk from a PostgreSQL client, or: psql -c "\copy (SELECT encode(content, 'base64') ...)"
CREATE TABLE IF NOT EXISTS tsmini.file (
    path    TEXT PRIMARY KEY,
    size    BIGINT NOT NULL,
    content BYTEA NOT NULL
);

-- Top-k most similar trajectories to trajectory p_idx by embedding L1 distance (what search_rome.py query does).
-- Example: SELECT * FROM tsmini.similar('rome_20200', 5);
CREATE OR REPLACE FUNCTION tsmini.similar(p_dataset TEXT, p_idx INT, k INT DEFAULT 10, p_model TEXT DEFAULT NULL)
RETURNS TABLE (rank BIGINT, idx INT, l1_distance REAL, geom_wgs84 geometry)
LANGUAGE sql STABLE AS $$
    WITH q AS (
        SELECT model, embedding FROM tsmini.embedding
        WHERE dataset = p_dataset AND idx = p_idx AND (p_model IS NULL OR model = p_model)
        ORDER BY model LIMIT 1
    ), d AS (
        SELECT e.idx,
               (SELECT sum(abs(a - b)) FROM unnest(e.embedding, q.embedding) AS u(a, b))::REAL AS l1
        FROM tsmini.embedding e JOIN q ON e.model = q.model
        WHERE e.dataset = p_dataset AND e.idx <> p_idx
        ORDER BY l1 LIMIT k
    )
    SELECT row_number() OVER (ORDER BY d.l1), d.idx, d.l1, t.geom_wgs84
    FROM d JOIN data.trajectory t ON t.dataset = p_dataset AND t.idx = d.idx
    ORDER BY d.l1
$$;
