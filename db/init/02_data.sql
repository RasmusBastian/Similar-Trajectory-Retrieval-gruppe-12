-- Shared trajectory data and ground truth, used by all model implementations.
CREATE SCHEMA IF NOT EXISTS data;

CREATE TABLE IF NOT EXISTS data.dataset (
    name       TEXT PRIMARY KEY,
    n          INT NOT NULL,
    merc_range DOUBLE PRECISION[] NOT NULL,
    source     TEXT
);

CREATE TABLE IF NOT EXISTS data.trajectory (
    dataset        TEXT NOT NULL REFERENCES data.dataset ON DELETE CASCADE,
    idx            INT NOT NULL,
    split          TEXT NOT NULL CHECK (split IN ('train', 'eval', 'test', 'unused')),
    n_points       INT NOT NULL,
    geom           geometry(LineString, 3857) NOT NULL,
    geom_wgs84     geometry(LineString, 4326) GENERATED ALWAYS AS (ST_Transform(geom, 4326)) STORED,
    PRIMARY KEY (dataset, idx)
);
CREATE INDEX IF NOT EXISTS trajectory_geom_wgs84_idx ON data.trajectory USING gist (geom_wgs84);

CREATE TABLE IF NOT EXISTS data.ground_truth_set (
    dataset      TEXT NOT NULL REFERENCES data.dataset ON DELETE CASCADE,
    measure      TEXT NOT NULL,
    max_distance DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (dataset, measure)
);

CREATE TABLE IF NOT EXISTS data.ground_truth (
    dataset   TEXT NOT NULL,
    measure   TEXT NOT NULL,
    split     TEXT NOT NULL CHECK (split IN ('train', 'eval', 'test')),
    row       INT NOT NULL,
    idx       INT NOT NULL,
    distances REAL[] NOT NULL,
    PRIMARY KEY (dataset, measure, split, row),
    FOREIGN KEY (dataset, measure) REFERENCES data.ground_truth_set ON DELETE CASCADE,
    FOREIGN KEY (dataset, idx) REFERENCES data.trajectory
);
