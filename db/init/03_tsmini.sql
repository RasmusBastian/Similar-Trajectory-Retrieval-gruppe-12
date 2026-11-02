-- TSMini (trajectory similarity learning, see TSMini/) on the Rome dataset: inputs and outputs.
-- Filled by TSMini/import_to_db.py (docker compose run --rm --build tsmini-import).
-- Also run by that script, so it must stay re-runnable (IF NOT EXISTS / OR REPLACE).

CREATE SCHEMA IF NOT EXISTS tsmini;

-- Input: map-matched Rome trajectories as delivered (matching_result.csv), before any filtering
CREATE TABLE IF NOT EXISTS tsmini.rome_raw_trajectory (
    traj_id  INT PRIMARY KEY,
    node_ids BIGINT[] NOT NULL,                  -- Node_list
    n_points INT NOT NULL,
    geom     geometry(LineString, 4326)          -- Coor_list, NULL if it has fewer than 2 points
);
CREATE INDEX IF NOT EXISTS rome_raw_trajectory_geom_idx ON tsmini.rome_raw_trajectory USING gist (geom);

-- Input: a dataset in TSMini's format (data/<name>.h5), made by TSMini/prepare_rome.py
CREATE TABLE IF NOT EXISTS tsmini.dataset (
    name       TEXT PRIMARY KEY,                 -- e.g. rome_20200
    n          INT NOT NULL,                     -- number of trajectories
    merc_range DOUBLE PRECISION[] NOT NULL,      -- [x_min, x_max, y_min, y_max] in Web Mercator metres
    source     TEXT
);

-- One row per trajectory in the .h5 file. idx is TSMini's index (/trajs_merc/<idx>).
-- split: which part of TSMini's 10,000-trajectory training slice it is in (train/eval/test), or 'unused'.
CREATE TABLE IF NOT EXISTS tsmini.trajectory (
    dataset        TEXT NOT NULL REFERENCES tsmini.dataset ON DELETE CASCADE,
    idx            INT NOT NULL,
    source_traj_id INT,                          -- traj_id in rome_raw_trajectory, NULL if unknown
    split          TEXT NOT NULL CHECK (split IN ('train', 'eval', 'test', 'unused')),
    n_points       INT NOT NULL,
    geom           geometry(LineString, 3857) NOT NULL,  -- exactly the points TSMini sees (Web Mercator)
    geom_wgs84     geometry(LineString, 4326) GENERATED ALWAYS AS (ST_Transform(geom, 4326)) STORED,
    PRIMARY KEY (dataset, idx)
);
CREATE INDEX IF NOT EXISTS trajectory_geom_wgs84_idx ON tsmini.trajectory USING gist (geom_wgs84);

-- Input: ground-truth distances TSMini trains on (data/<dataset>_traj_simi_dict_<measure>.pkl)
CREATE TABLE IF NOT EXISTS tsmini.ground_truth_set (
    dataset      TEXT NOT NULL REFERENCES tsmini.dataset ON DELETE CASCADE,
    measure      TEXT NOT NULL,                  -- frechet, dtw, hausdorff
    max_distance DOUBLE PRECISION NOT NULL,      -- TSMini divides by this to normalise
    PRIMARY KEY (dataset, measure)
);

-- One row per trajectory of a split; distances[j + 1] is the distance (metres) to row j of the same split.
-- The full symmetric matrix is stored (the .pkl only has the upper triangle).
CREATE TABLE IF NOT EXISTS tsmini.ground_truth (
    dataset   TEXT NOT NULL,
    measure   TEXT NOT NULL,
    split     TEXT NOT NULL CHECK (split IN ('train', 'eval', 'test')),
    row       INT NOT NULL,                      -- 0-based position within the split
    idx       INT NOT NULL,                      -- trajectory.idx
    distances REAL[] NOT NULL,
    PRIMARY KEY (dataset, measure, split, row),
    FOREIGN KEY (dataset, measure) REFERENCES tsmini.ground_truth_set ON DELETE CASCADE,
    FOREIGN KEY (dataset, idx) REFERENCES tsmini.trajectory
);

-- Output: 128-dimensional embedding of every trajectory (exp/embeddings_<dataset>.npz).
-- The L1 distance between two embeddings approximates the measure the model was trained on.
CREATE TABLE IF NOT EXISTS tsmini.embedding (
    dataset   TEXT NOT NULL,
    model     TEXT NOT NULL,                     -- checkpoint file name in tsmini.file
    idx       INT NOT NULL,
    embedding REAL[] NOT NULL,
    PRIMARY KEY (dataset, model, idx),
    FOREIGN KEY (dataset, idx) REFERENCES tsmini.trajectory ON DELETE CASCADE
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
    FROM d JOIN tsmini.trajectory t ON t.dataset = p_dataset AND t.idx = d.idx
    ORDER BY d.l1
$$;
