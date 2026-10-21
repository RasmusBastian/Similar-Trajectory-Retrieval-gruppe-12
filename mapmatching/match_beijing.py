"""Map-match Beijing trajectories with last year's map matching (FMM STMatch, see legacy/telchar) and store them.

Input is read from the shared aSTEP database (astep-aau/data): the road network (osm_edges) and the trips
(trajectories). The Beijing trips only contain a start and an end GPS point, plus the road segments the
trip drove on (road_points, already map-matched by the dataset authors). We therefore simulate a GPS trace
by sampling points along those road segments, map-match the trace with FMM, and compare the matched path
with the known road segments. Results go to bj_matched_trajectory in our own database.

Runs as the `mapmatching` service in docker-compose.yml: docker compose run --rm mapmatching --limit 1000
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import psycopg
import shapely
from pyproj import Transformer

from fmm import STMATCH, GPSConfig, Network, NetworkGraph, ResultConfig, STMATCHConfig

SCHEMA_SQL = Path(os.environ.get("SCHEMA_SQL", "/app/sql/02_beijing_map_matching.sql"))
METRIC_CRS = "EPSG:32650"  # UTM zone 50N, covers Beijing

# STMatch parameters are in the units of the network, and the Beijing network is in degrees (EPSG:4326).
CONFIGS = {
    # What legacy/telchar/main.py uses: the FMM defaults (300 radius, 50 GPS error, 30 max speed), meant for meters
    "legacy": lambda: STMATCHConfig(),
    # The same values in degrees: about 300 m radius, 50 m GPS error and 30 m/s max speed
    "degrees": lambda: STMATCHConfig(8, 0.003, 0.0005, 0.0003, 1.5),
}

UPSERT = """
    INSERT INTO bj_matched_trajectory (dataset, trajectory_id, config, taxi_id, start_time, end_time,
                                       true_edges, matched_edges, precision, recall, gps_geom, matched_geom)
    VALUES (%s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s), %s, %s, %s, %s,
            ST_GeomFromText(%s, 4326), ST_GeomFromText(%s, 4326))
    ON CONFLICT (dataset, trajectory_id, config) DO UPDATE SET
        taxi_id = EXCLUDED.taxi_id, start_time = EXCLUDED.start_time, end_time = EXCLUDED.end_time,
        true_edges = EXCLUDED.true_edges, matched_edges = EXCLUDED.matched_edges,
        precision = EXCLUDED.precision, recall = EXCLUDED.recall,
        gps_geom = EXCLUDED.gps_geom, matched_geom = EXCLUDED.matched_geom
"""


def connect_shared() -> psycopg.Connection:
    # Accepts the SQLAlchemy style URL from astep-aau/data's .env as well
    url = os.environ.get("SHARED_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/data")
    return psycopg.connect(url.replace("+psycopg2", ""))


def connect_own() -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5433"),
        user=os.environ.get("PGUSER", "str"),
        password=os.environ.get("PGPASSWORD", "str"),
        dbname=os.environ.get("PGDATABASE", "trajectories"),
    )


def load_network(shared: psycopg.Connection, shapefile: Path) -> dict[int, shapely.LineString]:
    """Writes osm_edges as a shapefile (FMM reads its network from a file) and returns the edges in meters."""
    rows = shared.execute("SELECT fid, u, v, ST_AsBinary(geometry) FROM osm_edges").fetchall()
    fid, u, v, wkb = zip(*rows)
    edges = gpd.GeoDataFrame(
        {"fid": fid, "u": u, "v": v}, geometry=shapely.from_wkb([bytes(b) for b in wkb]), crs="EPSG:4326"
    )
    edges.to_file(shapefile)
    print(f"  osm_edges: {len(edges)} edges")
    return dict(zip(edges["fid"], edges.geometry.to_crs(METRIC_CRS)))


def simulate_gps(road_points: list, edges_m: dict, interval: float, noise: float, rng) -> list[tuple]:
    """GPS points (x, y in meters, timestamp) every `interval` seconds along the trip's road segments.

    road_points elements are [edge_id, enter_timestamp, avg_speed, duration]. Each point gets Gaussian noise
    with standard deviation `noise` meters, like a real GPS receiver.
    """
    enter = np.array([p[1] for p in road_points], dtype=float)
    duration = np.array([p[3] for p in road_points], dtype=float)
    end = enter[-1] + duration[-1]
    points = []
    for t in np.append(np.arange(enter[0], end, interval), end):
        i = np.searchsorted(enter, t, side="right") - 1
        if duration[i] <= 0 or t > enter[i] + duration[i]:
            continue
        p = edges_m[road_points[i][0]].interpolate((t - enter[i]) / duration[i], normalized=True)
        points.append((p.x + rng.normal(0, noise), p.y + rng.normal(0, noise), t))
    return points


def run_stmatch(model: STMATCH, gps_file: Path, result_file: Path, config: STMATCHConfig) -> pd.DataFrame:
    """Same FMM calls as process_single_day in legacy/telchar/main.py."""
    gps_config = GPSConfig()
    gps_config.file = str(gps_file)
    gps_config.id = "id"
    gps_config.x = "x"
    gps_config.y = "y"
    gps_config.timestamp = "timestamp"
    gps_config.gps_point = True

    result_config = ResultConfig()
    result_config.file = str(result_file)

    model.match_gps_file(gps_config, result_config, config)
    return pd.read_csv(result_file, sep=";", dtype=str, keep_default_na=False).set_index("id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="traj_test", help="trajectories.dataset in the shared database")
    parser.add_argument("--limit", type=int, default=1000, help="number of trajectories to match")
    parser.add_argument("--interval", type=float, default=15, help="seconds between simulated GPS points")
    parser.add_argument("--noise", type=float, default=10, help="GPS noise standard deviation in meters")
    parser.add_argument("--configs", default="legacy,degrees", help=f"STMatch parameter sets: {','.join(CONFIGS)}")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    to_lonlat = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)

    with tempfile.TemporaryDirectory() as tmp, connect_shared() as shared:
        tmp = Path(tmp)
        print("Reading from the shared database ...")
        edges_m = load_network(shared, tmp / "edges.shp")
        trips = shared.execute(
            "SELECT taxi_id, trajectory_id, road_points FROM trajectories WHERE dataset = %s "
            "ORDER BY taxi_id, trajectory_id LIMIT %s",
            (args.dataset, args.limit),
        ).fetchall()
        print(f"  trajectories: {len(trips)} from {args.dataset}")

        # FMM needs integer trip ids, so trip i in `trips` gets id i
        gps_rows, gps_wkt, skipped = [], {}, 0
        for i, (_, _, road_points) in enumerate(trips):
            if not road_points or any(p[0] not in edges_m for p in road_points):
                skipped += 1
                continue
            points = simulate_gps(road_points, edges_m, args.interval, args.noise, rng)
            if len(points) < 2:
                skipped += 1
                continue
            lon, lat = to_lonlat.transform([p[0] for p in points], [p[1] for p in points])
            # Seconds since the trip started: FMM's point CSV reader parses timestamps with std::stof (32-bit
            # float), which rounds Unix timestamps to 128 s, so points 15 s apart would get the same time
            gps_rows += [(i, x, y, p[2] - points[0][2]) for x, y, p in zip(lon, lat, points)]
            gps_wkt[i] = shapely.LineString(zip(lon, lat)).wkt
        gps_file = tmp / "gps.csv"
        pd.DataFrame(gps_rows, columns=["id", "x", "y", "timestamp"]).to_csv(gps_file, sep=";", index=False)
        print(f"  simulated {len(gps_rows)} GPS points for {len(gps_wkt)} trips ({skipped} skipped)")

        print("Loading the network into FMM ...")
        network = Network(str(tmp / "edges.shp"), "fid", "u", "v")
        graph = NetworkGraph(network)  # STMATCH only keeps a reference, so the graph must stay alive
        model = STMATCH(network, graph)

        with connect_own() as own:
            own.execute(SCHEMA_SQL.read_text())
            for name in args.configs.split(","):
                started = time.time()
                result = run_stmatch(model, gps_file, tmp / f"result_{name}.csv", CONFIGS[name]())
                rows, precisions, recalls, exact = [], [], [], 0
                for i, wkt in gps_wkt.items():
                    taxi_id, trajectory_id, road_points = trips[i]
                    true_edges = [int(p[0]) for p in road_points]
                    cpath = result.loc[str(i), "cpath"] if str(i) in result.index else ""
                    matched = [int(e) for e in cpath.split(",")] if cpath else []
                    precision = recall = None
                    if matched:
                        common = set(matched) & set(true_edges)
                        precision = len(common) / len(set(matched))
                        recall = len(common) / len(set(true_edges))
                        precisions.append(precision)
                        recalls.append(recall)
                        exact += matched == true_edges
                    rows.append((
                        args.dataset, trajectory_id, name, taxi_id,
                        road_points[0][1], road_points[-1][1] + road_points[-1][3],
                        true_edges, matched, precision, recall,
                        wkt, result.loc[str(i), "mgeom"] if matched else None,
                    ))
                with own.cursor() as cur:
                    cur.executemany(UPSERT, rows)
                own.commit()

                n = len(rows)
                c = CONFIGS[name]()
                print(f"\n=== {name}: k={c.k} radius={c.radius} gps_error={c.gps_error} vmax={c.vmax}")
                print(f"  matched:        {len(precisions)}/{n} trips ({len(precisions) / n:.1%})")
                if precisions:
                    print(f"  mean precision: {np.mean(precisions):.1%}  (matched edges that are correct)")
                    print(f"  mean recall:    {np.mean(recalls):.1%}  (true edges that were found)")
                    print(f"  exact path:     {exact / n:.1%}")
                print(f"  time:           {time.time() - started:.0f} s, saved {n} rows to bj_matched_trajectory")


if __name__ == "__main__":
    main()
