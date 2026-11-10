#!/usr/bin/env python3
"""
Build TSMini's Beijing dataset from the trips in the shared aSTEP database (astep-aau/data),
in the same format as prepare_rome.py:

    data/beijing_20200.h5                            trajectories (Web-Mercator metres)
    data/beijing_20200_traj_simi_dict_<measure>.pkl  ground-truth distance matrices

TrajCL/prepare_rome.py --dataset beijing reads these files, so both models use the same
trajectories, splits and ground truth.

The Beijing trips have no GPS traces, only a start and an end point plus the road segments
they drove on (road_points, already map-matched by the DRPK authors). Like
mapmatching/match_beijing.py, we simulate a GPS point every 15 s along those road segments,
with 10 m Gaussian noise. A trip is kept if all its points are inside AREA and it has
20..200 points. The h5 file also stores each trajectory's id in the shared database
(/trajectory_id), which TSMini and TrajCL ignore.

Usage (inside the TSMini folder, venv active, shared database running on localhost:5432):
    python prepare_beijing.py                             # Frechet ground truth
    python prepare_beijing.py --measures frechet hausdorff
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import psycopg
import shapely
from pyproj import Transformer

from prepare_rome import MEASURES, N_TOTAL, lonlat_to_merc, write_ground_truth, write_h5

UTM_CRS = "EPSG:32650"  # UTM zone 50N, covers Beijing; the GPS noise is added in these metres

# Inside Beijing's 5th ring road, which holds 87% of the trips. The whole road network is about
# 170 x 180 km, which would give TrajCL millions of 100 m cells to embed.
# Must match the 'beijing' area in TrajCL/config.py.
AREA = (116.20, 39.75, 116.55, 40.03)  # min_lon, min_lat, max_lon, max_lat


def connect_shared():
    # Accepts the SQLAlchemy style URL from astep-aau/data's .env as well
    url = os.environ.get("SHARED_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/data")
    return psycopg.connect(url.replace("+psycopg2", ""))


def load_edges(conn):
    """Road segments that touch AREA, as LineStrings in UTM metres, keyed by osm_edges.fid."""
    rows = conn.execute(
        "SELECT fid, ST_AsBinary(geometry) FROM osm_edges "
        "WHERE geometry && ST_MakeEnvelope(%s, %s, %s, %s, 4326)",
        AREA,
    ).fetchall()
    fids, wkbs = zip(*rows)
    to_utm = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    lines = shapely.transform(
        shapely.from_wkb([bytes(b) for b in wkbs]),
        lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1])),
    )
    print(f"read {len(lines)} road segments")
    return dict(zip(fids, lines))


# Copied from mapmatching/match_beijing.py, which can't be imported here as it needs FMM
def simulate_gps(road_points, edges_m, interval, noise, rng):
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


def load_trajs(conn, edges, args):
    """The first args.n trips, in md5(trajectory_id) order, that pass the filters."""
    to_merc = Transformer.from_crs(UTM_CRS, "EPSG:3857", always_xy=True)  # same formula as lonlat_to_merc
    x_min, y_min = lonlat_to_merc(AREA[0], AREA[1])
    x_max, y_max = lonlat_to_merc(AREA[2], AREA[3])
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    trajs, ids, n_read = [], [], 0
    # md5 gives a fixed order that is spread over all taxis and days, so no shuffle is needed
    with conn.cursor(name="trips") as cur:
        cur.execute(
            "SELECT trajectory_id, road_points FROM trajectories WHERE dataset = %s "
            "ORDER BY md5(trajectory_id)",
            (args.dataset,),
        )
        for trajectory_id, road_points in cur:
            n_read += 1
            if not road_points or any(p[0] not in edges for p in road_points):
                continue
            points = simulate_gps(road_points, edges, args.interval, args.noise, rng)
            if len(points) < args.min_len:
                continue
            x, y = to_merc.transform([p[0] for p in points], [p[1] for p in points])
            pts = np.column_stack([x, y])
            # same rule as TSMini's remove_duplicate_points (consecutive equal x)
            pts = pts[np.insert(np.diff(pts[:, 0]) != 0, 0, True)]
            inside = (x_min <= pts[:, 0]).all() and (pts[:, 0] <= x_max).all() \
                and (y_min <= pts[:, 1]).all() and (pts[:, 1] <= y_max).all()
            if inside and args.min_len <= len(pts) <= args.max_len:
                trajs.append(pts)
                ids.append(trajectory_id)
                if len(trajs) == args.n:
                    break
    print(f"read {n_read} trips from {args.dataset}, kept {len(trajs)} in {time.time() - t0:.0f}s")
    return trajs, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="traj_test", help="trajectories.dataset in the shared database")
    # TrajCL pre-trains on the first 70%, and its paper finds little gain beyond 50,000 trajectories
    ap.add_argument("--n", type=int, default=75000, help="number of trajectories to keep")
    ap.add_argument("--interval", type=float, default=15, help="seconds between simulated GPS points")
    ap.add_argument("--noise", type=float, default=10, help="GPS noise standard deviation in metres")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--prefix", default="beijing_20200")
    ap.add_argument("--measures", nargs="+", default=["frechet"], choices=list(MEASURES))
    ap.add_argument("--min_len", type=int, default=20)   # Config.min_traj_len
    ap.add_argument("--max_len", type=int, default=200)  # Config.max_traj_len
    ap.add_argument("--seed", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with connect_shared() as conn:
        edges = load_edges(conn)
        trajs, ids = load_trajs(conn, edges, args)
    need = int(N_TOTAL / 0.3) + 1  # TSMini reads 10000 trajs starting at int(0.7*n)
    if len(trajs) < need:
        sys.exit(f"need at least {need} trajectories, only have {len(trajs)}")

    h5_path = os.path.join(args.out_dir, args.prefix + ".h5")
    write_h5(trajs, h5_path)
    with h5py.File(h5_path, "a") as f:
        f.create_dataset("trajectory_id", data=ids, dtype=h5py.string_dtype())
    write_ground_truth(trajs, args.measures, args.out_dir, args.prefix)
    print("done")


if __name__ == "__main__":
    main()
