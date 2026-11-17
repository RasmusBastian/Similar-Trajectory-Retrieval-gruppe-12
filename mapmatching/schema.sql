-- Optional output table for the separate map-matching experiment.
-- This file is deliberately outside db/init, so the project database does not create it on startup.
CREATE TABLE IF NOT EXISTS bj_matched_trajectory (
    dataset       TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    config        TEXT NOT NULL,
    taxi_id       TEXT NOT NULL,
    start_time    TIMESTAMPTZ NOT NULL,
    end_time      TIMESTAMPTZ NOT NULL,
    true_edges    INT[] NOT NULL,
    matched_edges INT[] NOT NULL,
    precision     REAL,
    recall        REAL,
    gps_geom      geometry(LineString, 4326),
    matched_geom  geometry(LineString, 4326),
    PRIMARY KEY (dataset, trajectory_id, config)
);
