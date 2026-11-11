#!/usr/bin/env python3
"""
Export map-matched Beijing trajectories from the shared aSTEP database (astep-aau/data)
into a CSV with the same columns as Rome's matching_result.csv, so prepare_rome.py can
turn it into TSMini's (and from there TrajCL's) input files unchanged.

The database's gps_points mostly have just 2 points (start and end), so trajectories are
built from road_points instead: the route's edges (osm_edges.fid) are replaced by their
end nodes (osm_nodes), like Rome's Node_list / Coor_list.

Only trajectories that lie completely inside --bbox (default: central Beijing, roughly
the 5th ring road) and have --min_len..--max_len points are kept. The bbox must lie
inside the 'beijing' area in ../TrajCL/config.py, whose cell space is built from it.
The sample is random but repeatable (ordered by a hash of --seed and trajectory_id).

Run with Docker (from the repository root, the shared database must be running):
    docker compose run --rm --build beijing-export                 # 50000 trajectories
    docker compose run --rm --build beijing-export --limit 100000
or with a venv that has psycopg:
    python export_beijing.py --db postgresql://postgres:postgres@localhost:5434/data

Then (inside TSMini, see readme.md):
    python prepare_rome.py --matching data/beijing_matching_result.csv --prefix beijing_20200
"""
import argparse
import csv
import json
import os
import time

import psycopg

# central Beijing, about 30 x 31 km (Rome's area in TrajCL/config.py is about 21 x 23 km)
DEFAULT_BBOX = [116.20, 39.75, 116.55, 40.03]  # min_lon min_lat max_lon max_lat


def load_network(cur):
    cur.execute("SELECT osmid, x, y FROM osm_nodes")
    nodes = {osmid: (x, y) for osmid, x, y in cur}
    cur.execute("SELECT fid, u, v FROM osm_edges")
    edges = {fid: (u, v) for fid, u, v in cur}
    print(f"road network: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


def route_nodes(road_points, edges):
    """Node sequence of a route: start and end node of every edge, without repeats.
    Consecutive edges share a node; if map matching left a gap, both ends are kept."""
    seq = []
    for point in road_points:  # [edge fid, time, speed, ...]
        u, v = edges[point[0]]
        for n in (u, v):
            if not seq or seq[-1] != n:
                seq.append(n)
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get(
        "SHARED_DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/data"))
    ap.add_argument("--dataset", default="traj_train_100",
                    help="trajectories.dataset to sample from (traj_train_100 contains all smaller train sets)")
    ap.add_argument("--out", default="data/beijing_matching_result.csv")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--min_len", type=int, default=20)   # Config.min_traj_len
    ap.add_argument("--max_len", type=int, default=200)  # Config.max_traj_len
    ap.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--seed", default="2000")
    args = ap.parse_args()
    min_lon, min_lat, max_lon, max_lat = args.bbox

    t0 = time.time()
    with psycopg.connect(args.db) as conn:
        with conn.cursor() as cur:
            nodes, edges = load_network(cur)

        # n nodes need at least n-1 edges and at most 2n, so filter roughly in SQL first
        with conn.cursor(name="trajs") as cur, open(args.out, "w", newline="") as fh:
            cur.itersize = 5000
            cur.execute(
                """
                SELECT taxi_id, trajectory_id, road_points FROM trajectories
                WHERE dataset = %s AND jsonb_array_length(road_points) BETWEEN %s AND %s
                ORDER BY md5(%s || trajectory_id)
                """,
                (args.dataset, (args.min_len + 1) // 2, args.max_len - 1, args.seed),
            )
            writer = csv.writer(fh)
            writer.writerow(["Traj_id", "Taxi_id", "Trajectory_id", "Node_list", "Coor_list"])
            seen = kept = 0
            for taxi_id, trajectory_id, road_points in cur:
                seen += 1
                seq = route_nodes(road_points, edges)
                if not args.min_len <= len(seq) <= args.max_len:
                    continue
                coords = [nodes[n] for n in seq]
                if not all(min_lon < x < max_lon and min_lat < y < max_lat for x, y in coords):
                    continue
                writer.writerow([kept, taxi_id, trajectory_id, json.dumps(seq),
                                 json.dumps([[x, y] for x, y in coords])])
                kept += 1
                if kept == args.limit:
                    break

    print(f"checked {seen} trajectories from {args.dataset}, kept {kept} "
          f"in {time.time() - t0:.0f}s -> {args.out}")
    if kept < args.limit:
        print(f"WARNING: only {kept} of the requested {args.limit} trajectories qualify")


if __name__ == "__main__":
    main()
