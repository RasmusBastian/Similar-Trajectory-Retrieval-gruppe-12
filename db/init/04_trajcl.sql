-- TrajCL (contrastive trajectory similarity learning, see TrajCL/): trained models and outputs.
-- Filled by TrajCL/import_to_db.py (docker compose run --rm --build trajcl-import).
-- Also run by that script, so it must stay re-runnable (IF NOT EXISTS / OR REPLACE).
--
-- TrajCL is trained on exactly the same trajectories, splits and ground truth as TSMini
-- (TrajCL/prepare_rome.py and prepare_beijing.py copy them), so those live only in the tsmini schema:
-- trajcl.embedding.idx is tsmini.trajectory.idx.

CREATE SCHEMA IF NOT EXISTS trajcl;

-- Output: embedding of every trajectory from a fine-tuned model (exp/embeddings_<dataset>.npz, made by TrajCL/embed.py).
-- The L1 distance between two embeddings approximates the measure the model was fine-tuned on.
CREATE TABLE IF NOT EXISTS trajcl.embedding (
    dataset   TEXT NOT NULL,
    model     TEXT NOT NULL,                     -- checkpoint file name in trajcl.file
    idx       INT NOT NULL,
    embedding REAL[] NOT NULL,
    PRIMARY KEY (dataset, model, idx),
    FOREIGN KEY (dataset, idx) REFERENCES tsmini.trajectory ON DELETE CASCADE
);

-- Output: training logs (exp/log/*.log)
CREATE TABLE IF NOT EXISTS trajcl.training_log (
    file_name TEXT PRIMARY KEY,
    content   TEXT NOT NULL
);

-- Output: everything that is expensive to recompute, path relative to TrajCL/:
--   data/<dataset>_cell100_cellspace.pkl, data/<dataset>_cell100_embdim*_embs.pkl   cell space + node2vec cell embeddings
--   exp/snapshots/<dataset>_node2vec_*.pt, <dataset>_TrajCL_best.pt                 node2vec and pre-trained model
--   exp/snapshots/<dataset>_trajsimi_*.pt                                          fine-tuned model
-- Written back to disk by: docker compose run --rm --build trajcl-import --restore
CREATE TABLE IF NOT EXISTS trajcl.file (
    path        TEXT PRIMARY KEY,
    size        BIGINT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    content     BYTEA NOT NULL
);

-- Top-k most similar trajectories to trajectory p_idx by embedding L1 distance.
-- Example: SELECT * FROM trajcl.similar('beijing_20200', 5);
CREATE OR REPLACE FUNCTION trajcl.similar(p_dataset TEXT, p_idx INT, k INT DEFAULT 10, p_model TEXT DEFAULT NULL)
RETURNS TABLE (rank BIGINT, idx INT, l1_distance REAL, geom_wgs84 geometry)
LANGUAGE sql STABLE AS $$
    WITH q AS (
        SELECT model, embedding FROM trajcl.embedding
        WHERE dataset = p_dataset AND idx = p_idx AND (p_model IS NULL OR model = p_model)
        ORDER BY model LIMIT 1
    ), d AS (
        SELECT e.idx,
               (SELECT sum(abs(a - b)) FROM unnest(e.embedding, q.embedding) AS u(a, b))::REAL AS l1
        FROM trajcl.embedding e JOIN q ON e.model = q.model
        WHERE e.dataset = p_dataset AND e.idx <> p_idx
        ORDER BY l1 LIMIT k
    )
    SELECT row_number() OVER (ORDER BY d.l1), d.idx, d.l1, t.geom_wgs84
    FROM d JOIN tsmini.trajectory t ON t.dataset = p_dataset AND t.idx = d.idx
    ORDER BY d.l1
$$;
