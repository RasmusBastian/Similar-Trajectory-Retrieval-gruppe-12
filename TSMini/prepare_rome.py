#!/usr/bin/env python3
"""
Convert the Rome map-matched file (matching_result.csv) into the two files
TSMini's train_trajsimi.py reads:

    data/rome_20200.h5                          trajectories (Web-Mercator metres)
    data/rome_20200_traj_simi_dict_<measure>.pkl  ground-truth distance matrices

Layout was taken from TSMini's own loader (utils/data_loader.py):
  * h5 attrs: 'n' (number of trajectories), 'merc_range' = [x_min, x_max, y_min, y_max]
  * h5 datasets: /trajs_merc/<i>  -> array of shape (len, 2)
  * TSMini uses 10,000 trajectories starting at index int(0.7 * n):
        first 7000 = train, next 1000 = eval, last 2000 = test
  * pkl: (trains_simi, evals_simi, tests_simi, max_distance)
        Each matrix is stored upper-triangular (zeros on and below the diagonal),
        because TSMini computes (M + M.T) / max_distance.

Usage (inside the TSMini folder, venv active):
    uv pip install numba
    python prepare_rome.py --matching /path/to/matching_result.csv
"""
import argparse
import csv
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import h5py
import numba as nb
from numba import njit, prange

R_EARTH = 6378137.0  # Web Mercator sphere radius (metres)
N_TOTAL = 10000      # hard-coded in TSMini's read_trajsimi_traj_dataset
N_TRAIN, N_EVAL, N_TEST = 7000, 1000, 2000


# --------------------------------------------------------------------------- #
# Data loading / conversion
# --------------------------------------------------------------------------- #
def lonlat_to_merc(lon, lat):
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def load_trajs(path, min_len, max_len):
    csv.field_size_limit(sys.maxsize)
    trajs = []
    n_rows = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            coords = json.loads(row["Coor_list"])  # [[lon, lat], ...]
            pts = [lonlat_to_merc(lon, lat) for lon, lat in coords]
            # same rule as TSMini's remove_duplicate_points (consecutive equal x)
            pts = [p for i, p in enumerate(pts) if i == 0 or p[0] != pts[i - 1][0]]
            if min_len <= len(pts) <= max_len:
                trajs.append(np.asarray(pts, dtype=np.float64))
    print(f"read {n_rows} rows, kept {len(trajs)} with {min_len}..{max_len} points")
    return trajs


def write_h5(trajs, path):
    allpts = np.concatenate(trajs)
    merc_range = np.array(
        [allpts[:, 0].min(), allpts[:, 0].max(), allpts[:, 1].min(), allpts[:, 1].max()]
    )
    with h5py.File(path, "w") as f:
        f.attrs["n"] = len(trajs)
        f.attrs["merc_range"] = merc_range
        g = f.create_group("trajs_merc")
        for i, t in enumerate(trajs):
            g.create_dataset(str(i), data=t)
    print(f"wrote {path}  (n={len(trajs)}, merc_range={merc_range.tolist()})")


# --------------------------------------------------------------------------- #
# Distance measures (Euclidean in the Mercator plane, like the Porto pipeline)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _d(a, i, b, j):
    dx = a[i, 0] - b[j, 0]
    dy = a[i, 1] - b[j, 1]
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def frechet(a, la, b, lb):
    """Discrete Frechet distance."""
    prev = np.empty(lb)
    cur = np.empty(lb)
    prev[0] = _d(a, 0, b, 0)
    for j in range(1, lb):
        prev[j] = max(prev[j - 1], _d(a, 0, b, j))
    for i in range(1, la):
        cur[0] = max(prev[0], _d(a, i, b, 0))
        for j in range(1, lb):
            m = min(prev[j], prev[j - 1], cur[j - 1])
            cur[j] = max(m, _d(a, i, b, j))
        for j in range(lb):
            prev[j] = cur[j]
    return prev[lb - 1]


@njit(cache=True)
def dtw(a, la, b, lb):
    """Dynamic time warping (sum of Euclidean point distances)."""
    prev = np.empty(lb)
    cur = np.empty(lb)
    prev[0] = _d(a, 0, b, 0)
    for j in range(1, lb):
        prev[j] = prev[j - 1] + _d(a, 0, b, j)
    for i in range(1, la):
        cur[0] = prev[0] + _d(a, i, b, 0)
        for j in range(1, lb):
            cur[j] = _d(a, i, b, j) + min(prev[j], prev[j - 1], cur[j - 1])
        for j in range(lb):
            prev[j] = cur[j]
    return prev[lb - 1]


@njit(cache=True)
def hausdorff(a, la, b, lb):
    """Symmetric Hausdorff distance between the two point sets."""
    h_ab = 0.0
    for i in range(la):
        m = np.inf
        for j in range(lb):
            m = min(m, _d(a, i, b, j))
        h_ab = max(h_ab, m)
    h_ba = 0.0
    for j in range(lb):
        m = np.inf
        for i in range(la):
            m = min(m, _d(a, i, b, j))
        h_ba = max(h_ba, m)
    return max(h_ab, h_ba)


MEASURES = {"frechet": frechet, "dtw": dtw, "hausdorff": hausdorff}


def make_matrix_fn(measure):
    @njit(parallel=True, cache=True)
    def upper_matrix(P, L, order):
        n = P.shape[0]
        M = np.zeros((n, n), dtype=np.float32)
        for k in prange(n):
            i = order[k]
            for j in range(i + 1, n):
                M[i, j] = measure(P[i], L[i], P[j], L[j])
        return M

    return upper_matrix


def pad(trajs, max_len):
    P = np.zeros((len(trajs), max_len, 2), dtype=np.float64)
    L = np.zeros(len(trajs), dtype=np.int64)
    for i, t in enumerate(trajs):
        P[i, : len(t)] = t
        L[i] = len(t)
    return P, L


def balanced_order(n):
    # early rows have more pairs; interleave long and short rows so that
    # every thread gets a similar amount of work
    lo, hi = list(range(n // 2 + n % 2)), list(range(n - 1, n // 2 + n % 2 - 1, -1))
    out = []
    for a, b in zip(lo, hi):
        out += [a, b]
    out += lo[len(hi):]
    return np.asarray(out, dtype=np.int64)


def simi_matrix(fn, trajs, name):
    t0 = time.time()
    P, L = pad(trajs, max(len(t) for t in trajs))
    M = fn(P, L, balanced_order(len(trajs)))
    print(f"  {name}: {len(trajs)}x{len(trajs)} in {time.time() - t0:.1f}s")
    return M


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True, help="path to matching_result.csv")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--prefix", default="rome_20200")
    ap.add_argument("--measures", nargs="+", default=["frechet"], choices=list(MEASURES))
    ap.add_argument("--min_len", type=int, default=20)   # Config.min_traj_len
    ap.add_argument("--max_len", type=int, default=200)  # Config.max_traj_len
    ap.add_argument("--seed", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    trajs = load_trajs(args.matching, args.min_len, args.max_len)
    need = int(N_TOTAL / 0.3) + 1  # TSMini reads 10000 trajs starting at int(0.7*n)
    if len(trajs) < need:
        sys.exit(f"need at least {need} trajectories, only have {len(trajs)}")

    # shuffle so the train/eval/test slice is not one contiguous block of time
    rng = np.random.default_rng(args.seed)
    trajs = [trajs[i] for i in rng.permutation(len(trajs))]

    h5_path = os.path.join(args.out_dir, args.prefix + ".h5")
    write_h5(trajs, h5_path)

    off = int(len(trajs) * 0.7)
    used = trajs[off : off + N_TOTAL]
    splits = {
        "trains": used[:N_TRAIN],
        "evals": used[N_TRAIN : N_TRAIN + N_EVAL],
        "tests": used[N_TRAIN + N_EVAL :],
    }
    assert [len(v) for v in splits.values()] == [N_TRAIN, N_EVAL, N_TEST]

    for name in args.measures:
        print(f"computing {name} ground truth (first call compiles, ~20s)...")
        fn = make_matrix_fn(MEASURES[name])
        mats = {k: simi_matrix(fn, v, k) for k, v in splits.items()}
        max_distance = float(max(m.max() for m in mats.values()))
        out = os.path.join(args.out_dir, f"{args.prefix}_traj_simi_dict_{name}.pkl")
        with open(out, "wb") as fh:
            pickle.dump((mats["trains"], mats["evals"], mats["tests"], max_distance), fh,
                        protocol=4)
        print(f"wrote {out}  (max_distance={max_distance:.1f} m)")
    print("done")


if __name__ == "__main__":
    main()
