#!/usr/bin/env python3
"""
Put TSMini's inputs and outputs into the project database (schema tsmini, see db/init/03_tsmini.sql),
so they can be looked at with any PostgreSQL client and shared without passing files around.

    inputs   matching_result.csv (raw Rome trajectories)  -> tsmini.rome_raw_trajectory
             data/<prefix>.h5                             -> tsmini.dataset, tsmini.trajectory
             data/<prefix>_traj_simi_dict_<measure>.pkl   -> tsmini.ground_truth_set, tsmini.ground_truth
    outputs  exp/embeddings_<prefix>.npz                  -> tsmini.embedding
             exp/log/*.log                                -> tsmini.training_log
             exp/snapshots/*.pt, *.png                    -> tsmini.file

Normally run through Docker from the repository root (no local Python setup needed):

    docker compose run --rm --build tsmini-import              # import what is not there yet
    docker compose run --rm --build tsmini-import --replace    # delete and re-import everything

Missing files are skipped with a message. Connection settings come from the PG* environment variables.
"""
import argparse
import csv
import glob
import json
import math
import os
import pickle
import sys

import h5py
import numpy as np
import psycopg

SQL_SCHEMA = os.environ.get("TSMINI_SQL", "/app/sql/03_tsmini.sql")
R_EARTH = 6378137.0
N_TRAIN, N_EVAL, N_TEST = 7000, 1000, 2000  # TSMini's split of the 10,000 trajectories from int(0.7 * n)


def connect():
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5433"),
        user=os.environ.get("PGUSER", "str"),
        password=os.environ.get("PGPASSWORD", "str"),
        dbname=os.environ.get("PGDATABASE", "trajectories"),
    )


def exists(cur, sql, *params):
    cur.execute(f"SELECT EXISTS ({sql})", params)
    return cur.fetchone()[0]


def linestring(points, srid):
    return f"SRID={srid};LINESTRING({','.join(f'{x!r} {y!r}' for x, y in points)})"


# ----------------------------------------------------------------------------- inputs
def read_raw(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield int(row["Traj_id"]), json.loads(row["Node_list"]), json.loads(row["Coor_list"])


def import_raw(cur, path):
    n = 0
    with cur.copy("COPY tsmini.rome_raw_trajectory (traj_id, node_ids, n_points, geom) FROM STDIN") as copy:
        for traj_id, nodes, coords in read_raw(path):
            geom = linestring(coords, 4326) if len(coords) >= 2 else None
            copy.write_row((traj_id, nodes, len(coords), geom))
            n += 1
    print(f"  rome_raw_trajectory: {n} rows")


def source_ids(raw_path, trajs, min_len=20, max_len=200, seed=2000):
    """Recover which raw traj_id each .h5 trajectory came from by redoing prepare_rome.py's filtering
    and shuffle. Returns None if the result does not reproduce the .h5 exactly."""
    kept = []
    for traj_id, _, coords in read_raw(raw_path):
        pts = [(R_EARTH * math.radians(lon), R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))
               for lon, lat in coords]
        pts = [p for i, p in enumerate(pts) if i == 0 or p[0] != pts[i - 1][0]]
        if min_len <= len(pts) <= max_len:
            kept.append((traj_id, np.asarray(pts)))
    if len(kept) != len(trajs):
        return None
    order = np.random.default_rng(seed).permutation(len(kept))
    ids = []
    for t, k in zip(trajs, order):
        traj_id, pts = kept[k]
        if pts.shape != t.shape or not np.allclose(pts, t, rtol=0, atol=1e-6):
            return None
        ids.append(traj_id)
    return ids


def split_of(idx, n):
    off = int(n * 0.7)
    for name, lo, hi in (("train", 0, N_TRAIN), ("eval", N_TRAIN, N_TRAIN + N_EVAL),
                         ("test", N_TRAIN + N_EVAL, N_TRAIN + N_EVAL + N_TEST)):
        if off + lo <= idx < off + hi:
            return name
    return "unused"


def import_dataset(cur, prefix, h5_path, raw_path):
    with h5py.File(h5_path, "r") as f:
        n = int(f.attrs["n"])
        merc_range = [float(v) for v in f.attrs["merc_range"]]
        trajs = [f[f"/trajs_merc/{i}"][:] for i in range(n)]
    ids = source_ids(raw_path, trajs) if raw_path else None
    if raw_path and ids is None:
        print("  could not match .h5 trajectories to matching_result.csv, source_traj_id left empty")

    cur.execute("INSERT INTO tsmini.dataset (name, n, merc_range, source) VALUES (%s, %s, %s, %s)",
                (prefix, n, merc_range, "TSMini/prepare_rome.py on matching_result.csv"))
    with cur.copy("COPY tsmini.trajectory (dataset, idx, source_traj_id, split, n_points, geom) FROM STDIN") as copy:
        for i, t in enumerate(trajs):
            copy.write_row((prefix, i, ids[i] if ids else None, split_of(i, n), len(t), linestring(t.tolist(), 3857)))
    print(f"  dataset {prefix}: {n} trajectories" + (", linked to raw traj_id" if ids else ""))
    return n


def import_ground_truth(cur, prefix, measure, pkl_path, n):
    with open(pkl_path, "rb") as f:
        trains, evals, tests, max_distance = pickle.load(f)
    cur.execute("INSERT INTO tsmini.ground_truth_set (dataset, measure, max_distance) VALUES (%s, %s, %s)",
                (prefix, measure, float(max_distance)))
    off = int(n * 0.7)
    with cur.copy("COPY tsmini.ground_truth (dataset, measure, split, row, idx, distances) FROM STDIN "
                  "WITH (FORMAT binary)") as copy:
        copy.set_types(["text", "text", "text", "int4", "int4", "float4[]"])
        for split, upper, base in (("train", trains, 0), ("eval", evals, N_TRAIN), ("test", tests, N_TRAIN + N_EVAL)):
            full = np.asarray(upper, dtype=np.float32)
            full = full + full.T  # stored upper-triangular with zeros below
            for r, row in enumerate(full):
                copy.write_row((prefix, measure, split, r, off + base + r, row.tolist()))
            print(f"  ground_truth {measure} {split}: {len(full)}x{len(full)}")


# ----------------------------------------------------------------------------- outputs
def import_embeddings(cur, prefix, npz_path, model):
    emb = np.load(npz_path)["emb"]
    with cur.copy("COPY tsmini.embedding (dataset, model, idx, embedding) FROM STDIN WITH (FORMAT binary)") as copy:
        copy.set_types(["text", "text", "int4", "float4[]"])
        for i, e in enumerate(emb):
            copy.write_row((prefix, model, i, e.tolist()))
    print(f"  embedding: {emb.shape[0]} x {emb.shape[1]} (model {model})")


def import_logs(cur, root):
    for path in sorted(glob.glob(os.path.join(root, "exp/log/*.log"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            cur.execute("INSERT INTO tsmini.training_log (file_name, content) VALUES (%s, %s) "
                        "ON CONFLICT (file_name) DO UPDATE SET content = EXCLUDED.content",
                        (os.path.basename(path), f.read()))
        print(f"  training_log: {os.path.basename(path)}")


def import_files(cur, root):
    paths = sorted(glob.glob(os.path.join(root, "exp/snapshots/*.pt")) + glob.glob(os.path.join(root, "*.png")))
    for path in paths:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        with open(path, "rb") as f:
            data = f.read()
        cur.execute("INSERT INTO tsmini.file (path, size, content) VALUES (%s, %s, %s) "
                    "ON CONFLICT (path) DO UPDATE SET size = EXCLUDED.size, content = EXCLUDED.content",
                    (rel, len(data), data))
        print(f"  file: {rel} ({len(data) / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)), help="TSMini folder")
    ap.add_argument("--raw", default=os.environ.get("ROME_MATCHING_CSV", ""), help="path to matching_result.csv")
    ap.add_argument("--prefix", default="rome_20200")
    ap.add_argument("--replace", action="store_true", help="delete existing TSMini data first")
    args = ap.parse_args()

    root, prefix = args.root, args.prefix
    raw = args.raw if args.raw and os.path.isfile(args.raw) else None
    h5 = os.path.join(root, "data", prefix + ".h5")
    npz = os.path.join(root, "exp", f"embeddings_{prefix}.npz")
    ckpts = sorted(glob.glob(os.path.join(root, "exp/snapshots", prefix + "_*.pt")))

    with connect() as conn, conn.cursor() as cur:
        with open(SQL_SCHEMA) as f:
            cur.execute(f.read())
        if args.replace:
            cur.execute("TRUNCATE tsmini.rome_raw_trajectory, tsmini.dataset, tsmini.trajectory, "
                        "tsmini.ground_truth_set, tsmini.ground_truth, tsmini.embedding, "
                        "tsmini.training_log, tsmini.file")
            print("deleted existing TSMini data")

        print("inputs:")
        if not raw:
            print("  matching_result.csv not found, skipping raw trajectories (set ROME_MATCHING_CSV)")
        elif exists(cur, "SELECT 1 FROM tsmini.rome_raw_trajectory"):
            print("  rome_raw_trajectory already imported")
        else:
            import_raw(cur, raw)

        if exists(cur, "SELECT 1 FROM tsmini.dataset WHERE name = %s", prefix):
            print(f"  dataset {prefix} already imported")
        elif not os.path.isfile(h5):
            sys.exit(f"  {h5} not found, nothing else can be imported without it")
        else:
            import_dataset(cur, prefix, h5, raw)
        cur.execute("SELECT n FROM tsmini.dataset WHERE name = %s", (prefix,))
        n = cur.fetchone()[0]

        for pkl in sorted(glob.glob(os.path.join(root, "data", f"{prefix}_traj_simi_dict_*.pkl"))):
            measure = os.path.basename(pkl)[len(prefix + "_traj_simi_dict_"):-len(".pkl")]
            if exists(cur, "SELECT 1 FROM tsmini.ground_truth_set WHERE dataset = %s AND measure = %s", prefix, measure):
                print(f"  ground truth {measure} already imported")
            else:
                import_ground_truth(cur, prefix, measure, pkl, n)

        print("outputs:")
        model = os.path.basename(ckpts[-1]) if ckpts else "unknown"
        if not os.path.isfile(npz):
            print(f"  {npz} not found, skipping embeddings")
        elif exists(cur, "SELECT 1 FROM tsmini.embedding WHERE dataset = %s AND model = %s", prefix, model):
            print("  embeddings already imported")
        else:
            import_embeddings(cur, prefix, npz, model)
        import_logs(cur, root)
        import_files(cur, root)
        print("committing...")
    print("done")


if __name__ == "__main__":
    main()
