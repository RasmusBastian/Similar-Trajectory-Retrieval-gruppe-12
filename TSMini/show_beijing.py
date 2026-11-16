#!/usr/bin/env python3
"""
Draw the Beijing data from prepare_beijing.py, to see what TSMini and TrajCL get as input.

    python show_beijing.py trip --idx 0    # one trip: the roads it drove on and the simulated GPS points
    python show_beijing.py facit --idx 0   # test trip 0..1999 and its 5 most similar trips by Frechet and Hausdorff

Writes exp/beijing_trip_<idx>.png or exp/beijing_facit_<idx>.png. Needs the shared database on
localhost:5432 for the road network, like prepare_beijing.py.
"""
import argparse
import os
import pickle

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.collections import LineCollection
from pyproj import Transformer

from prepare_beijing import connect_shared
from prepare_rome import N_EVAL, N_TRAIN

TO_LONLAT = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
TO_MERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def to_merc(lines):
    return shapely.transform(lines, lambda xy: np.column_stack(TO_MERC.transform(xy[:, 0], xy[:, 1])))


def draw_roads(ax, conn, trajs, margin=300):
    """All road segments around `trajs` in light grey, as a background map."""
    pts = np.concatenate(trajs)
    x_min, y_min = pts.min(axis=0) - margin
    x_max, y_max = pts.max(axis=0) + margin
    lon_min, lat_min = TO_LONLAT.transform(x_min, y_min)
    lon_max, lat_max = TO_LONLAT.transform(x_max, y_max)
    rows = conn.execute(
        "SELECT ST_AsBinary(geometry) FROM osm_edges WHERE geometry && ST_MakeEnvelope(%s, %s, %s, %s, 4326)",
        (lon_min, lat_min, lon_max, lat_max),
    ).fetchall()
    lines = to_merc(shapely.from_wkb([bytes(r[0]) for r in rows]))
    ax.add_collection(LineCollection([shapely.get_coordinates(l) for l in lines], colors="0.85", linewidths=0.6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def show_trip(f, conn, idx, dataset):
    traj = f[f"/trajs_merc/{idx}"][:]
    trajectory_id = f["trajectory_id"][idx].decode()
    road_points = conn.execute(
        "SELECT road_points FROM trajectories WHERE dataset = %s AND trajectory_id = %s",
        (dataset, trajectory_id),
    ).fetchone()[0]
    fids = [p[0] for p in road_points]
    rows = dict(conn.execute(
        "SELECT fid, ST_AsBinary(geometry) FROM osm_edges WHERE fid = ANY(%s)", (fids,)
    ).fetchall())
    route = to_merc(shapely.from_wkb([bytes(rows[fid]) for fid in fids]))
    minutes = (road_points[-1][1] + road_points[-1][3] - road_points[0][1]) / 60

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_roads(ax, conn, [traj])
    ax.add_collection(LineCollection([shapely.get_coordinates(l) for l in route], colors="tab:blue",
                                     linewidths=4, alpha=0.5, label=f"roads driven ({len(fids)} segments)"))
    ax.plot(traj[:, 0], traj[:, 1], "-o", color="tab:red", markersize=4, linewidth=1,
            label=f"simulated GPS ({len(traj)} points, every 15 s)")
    ax.plot(*traj[0], "s", color="black", markersize=9, label="start")
    ax.set_title(f"Trip {idx} ({trajectory_id}), {minutes:.0f} min")
    ax.legend(loc="best")
    return fig


def show_facit(f, conn, idx, prefix):
    n = int(f.attrs["n"])
    first_test = int(n * 0.7) + N_TRAIN + N_EVAL  # TSMini's test trips start here, see prepare_rome.py
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, measure in zip(axes, ["frechet", "hausdorff"]):
        with open(f"data/{prefix}_traj_simi_dict_{measure}.pkl", "rb") as fh:
            tests = pickle.load(fh)[2]
        dist = tests + tests.T  # stored upper-triangular
        row = dist[idx].astype(float)
        row[idx] = np.inf
        top = np.argsort(row)[:5]
        query = f[f"/trajs_merc/{first_test + idx}"][:]
        neighbours = [f[f"/trajs_merc/{first_test + j}"][:] for j in top]

        draw_roads(ax, conn, [query, *neighbours])
        for rank, (j, t) in enumerate(zip(top, neighbours), 1):
            ax.plot(t[:, 0], t[:, 1], "-", linewidth=2, label=f"{rank}. test trip {j}: {row[j]:.0f} m")
            ax.plot(*t[0], "o", color=ax.lines[-1].get_color(), markersize=6)
        ax.plot(query[:, 0], query[:, 1], "-", color="black", linewidth=3.5, label=f"query: test trip {idx}")
        ax.plot(*query[0], "s", color="black", markersize=9)
        ax.set_title(f"5 most similar by {measure.capitalize()} (circle/square = start)")
        ax.legend(loc="best", fontsize=9)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["trip", "facit"])
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--prefix", default="beijing_20200")
    ap.add_argument("--dataset", default="traj_test", help="the --dataset prepare_beijing.py used")
    args = ap.parse_args()

    with h5py.File(f"data/{args.prefix}.h5", "r") as f, connect_shared() as conn:
        if args.what == "trip":
            fig = show_trip(f, conn, args.idx, args.dataset)
        else:
            fig = show_facit(f, conn, args.idx, args.prefix)
    os.makedirs("exp", exist_ok=True)
    out = f"exp/beijing_{args.what}_{args.idx}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
