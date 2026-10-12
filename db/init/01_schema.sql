-- Runs automatically the first time the database container starts (empty volume).
-- To re-run after changes: docker compose down -v && docker compose up

CREATE EXTENSION IF NOT EXISTS postgis;

-- Road network (from legacy/telchar/output)
CREATE TABLE vertex (
    node_id   BIGINT PRIMARY KEY,
    longitude DOUBLE PRECISION NOT NULL,
    latitude  DOUBLE PRECISION NOT NULL,
    geom      geometry(Point, 4326)
              GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED
);
CREATE INDEX vertex_geom_idx ON vertex USING gist (geom);

CREATE TABLE edge (
    edge_id    INT PRIMARY KEY,
    start_node BIGINT NOT NULL REFERENCES vertex,
    end_node   BIGINT NOT NULL REFERENCES vertex,
    geom       geometry(LineString, 4326) NOT NULL,
    length_m   DOUBLE PRECISION NOT NULL
);
CREATE INDEX edge_geom_idx ON edge USING gist (geom);
CREATE INDEX edge_start_node_idx ON edge (start_node);

-- Cleaned travel times (from legacy/postprocessing/output), one row per day and edge.
-- seconds[i] is the traversal time in 5-minute slot i (1-based: seconds[1] = 00:00-00:05, seconds[288] = 23:55-24:00).
CREATE TABLE edge_travel_time (
    day     SMALLINT NOT NULL,
    edge_id INT NOT NULL REFERENCES edge,
    seconds REAL[] NOT NULL,
    PRIMARY KEY (day, edge_id)
);

-- Trajectories as sequences of road segments (filled later: synthetic, map-matched, ...)
CREATE TABLE trajectory (
    trajectory_id TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    start_time    TIMESTAMPTZ,
    end_time      TIMESTAMPTZ
);

CREATE TABLE trajectory_edge (
    trajectory_id TEXT NOT NULL REFERENCES trajectory ON DELETE CASCADE,
    seq           INT NOT NULL,
    edge_id       INT NOT NULL REFERENCES edge,
    enter_time    TIMESTAMPTZ,
    PRIMARY KEY (trajectory_id, seq)
);
-- Inverted index edge -> trajectories, used for candidate generation in retrieval
CREATE INDEX trajectory_edge_edge_id_idx ON trajectory_edge (edge_id);
