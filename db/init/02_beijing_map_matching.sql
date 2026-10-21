-- Beijing trajectories map-matched with last year's map matching (FMM STMatch), see mapmatching/.
-- Edge ids are osm_edges.fid from the shared aSTEP database (astep-aau/data), not our Harbin edge table,
-- so there is no foreign key to edge.
-- Also run by mapmatching/match_beijing.py, so it must stay re-runnable (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS bj_matched_trajectory (
    dataset       TEXT NOT NULL,         -- e.g. traj_test
    trajectory_id TEXT NOT NULL,
    config        TEXT NOT NULL,         -- STMatch parameter set: 'legacy' or 'degrees'
    taxi_id       TEXT NOT NULL,
    start_time    TIMESTAMPTZ NOT NULL,
    end_time      TIMESTAMPTZ NOT NULL,
    true_edges    INT[] NOT NULL,        -- road_points from the dataset (ground truth)
    matched_edges INT[] NOT NULL,        -- FMM cpath, empty if matching failed
    precision     REAL,                  -- share of matched edges that are in true_edges
    recall        REAL,                  -- share of true_edges that were matched
    gps_geom      geometry(LineString, 4326),  -- simulated GPS trace given to FMM
    matched_geom  geometry(LineString, 4326),  -- FMM mgeom
    PRIMARY KEY (dataset, trajectory_id, config)
);
