#!/usr/bin/env python3
"""
Save TrajCL's trained models and outputs in the project database (schema trajcl, see db/init/04_trajcl.sql),
so nothing has to be trained twice, and get them back onto disk on another machine.

    data/<prefix>_cell*_cellspace.pkl, *_embs.pkl  -> trajcl.file   (node2vec cell embeddings, ~9 h on CPU)
    exp/snapshots/<prefix>_*.pt                    -> trajcl.file   (node2vec, pre-trained and fine-tuned models)
    exp/log/*.log                                  -> trajcl.training_log
    exp/embeddings_<prefix>.npz (embed.py)         -> trajcl.embedding

The trajectories, splits and ground truth are the same as TSMini's and are stored in the tsmini schema.
If that dataset is not in the database yet, it is imported from ../TSMini/data first.

Normally run through Docker from the repository root:

    docker compose run --rm --build trajcl-import --prefix beijing_20200            # save what is new or changed
    docker compose run --rm --build trajcl-import --prefix beijing_20200 --restore  # write the saved files back to disk

Connection settings come from the PG* environment variables.
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import psycopg

HERE = os.path.dirname(os.path.abspath(__file__))
SQL_DIR = os.environ.get("SQL_DIR", "/app/sql")
TSMINI_ROOT = os.environ.get("TSMINI_ROOT", os.path.join(HERE, "..", "TSMini"))


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


def ensure_tsmini_dataset(cur, prefix):
    """TrajCL's embeddings point at tsmini.trajectory, so make sure the dataset is there."""
    with open(os.path.join(SQL_DIR, "03_tsmini.sql")) as f:
        cur.execute(f.read())
    if exists(cur, "SELECT 1 FROM tsmini.dataset WHERE name = %s", prefix):
        return True
    h5 = os.path.join(TSMINI_ROOT, "data", prefix + ".h5")
    if not os.path.isfile(h5):
        print(f"  dataset {prefix} is not in tsmini.dataset and {h5} is missing, skipping embeddings")
        return False
    sys.path.insert(0, TSMINI_ROOT)
    from import_to_db import import_dataset, import_ground_truth  # TSMini's importer
    n = import_dataset(cur, prefix, h5)
    for pkl in sorted(glob.glob(os.path.join(TSMINI_ROOT, "data", f"{prefix}_traj_simi_dict_*.pkl"))):
        import_ground_truth(cur, prefix, os.path.basename(pkl)[len(prefix + "_traj_simi_dict_"):-len(".pkl")], pkl, n)
    return True


def saved_files(root, prefix):
    paths = (glob.glob(os.path.join(root, "data", f"{prefix}_cell*_cellspace.pkl"))
             + glob.glob(os.path.join(root, "data", f"{prefix}_cell*_embs.pkl"))
             + glob.glob(os.path.join(root, "exp", "snapshots", f"{prefix}_*.pt")))
    return sorted(path for path in paths if os.path.isfile(path))


def import_files(cur, root, prefix):
    for path in saved_files(root, prefix):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        size = os.path.getsize(path)
        if exists(cur, "SELECT 1 FROM trajcl.file WHERE path = %s AND size = %s", rel, size):
            print(f"  file: {rel} unchanged")
            continue
        with open(path, "rb") as f:
            data = f.read()
        cur.execute("INSERT INTO trajcl.file (path, size, content) VALUES (%s, %s, %s) "
                    "ON CONFLICT (path) DO UPDATE SET size = EXCLUDED.size, content = EXCLUDED.content, "
                    "imported_at = now()", (rel, size, data))
        print(f"  file: {rel} ({size / 1e6:.1f} MB)")


def import_test_inputs(cur, root, prefix):
    for path in sorted(glob.glob(os.path.join(root, "data", f"{prefix}_newsimi_*.pkl"))):
        name = os.path.basename(path)[len(prefix + "_newsimi_"):-len(".pkl")]
        if name == "raw":
            variant, rate = "raw", 0.0
        else:
            match = re.fullmatch(r"(downsampling|distort)_([0-9.]+)", name)
            if not match:
                print(f"  test input: unrecognized name {name}, skipping")
                continue
            variant, rate = match.group(1), float(match.group(2))
        with open(path, "rb") as f:
            queries, database = pickle.load(f)
        cur.execute("DELETE FROM trajcl.test_trajectory WHERE dataset = %s AND variant = %s AND rate = %s",
                    (prefix, variant, rate))
        with cur.copy("COPY trajcl.test_trajectory (dataset, variant, rate, role, idx, points) FROM STDIN WITH (FORMAT binary)") as copy:
            copy.set_types(["text", "text", "float8", "text", "int4", "float8[]"])
            for role, sequences in (("query", queries), ("database", database)):
                for idx, points in enumerate(sequences):
                    copy.write_row((prefix, variant, rate, role, idx, points))
        print(f"  test input: {variant} {rate:g} ({len(queries)} queries, {len(database)} database trajectories)")


def import_logs(cur, root, prefix):
    for path in sorted(glob.glob(os.path.join(root, "exp", "log", "*.log"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        if prefix not in content and prefix not in os.path.basename(path):
            continue
        cur.execute("INSERT INTO trajcl.training_log (file_name, content) VALUES (%s, %s) "
                        "ON CONFLICT (file_name) DO UPDATE SET content = EXCLUDED.content",
                    (os.path.basename(path), content))
        print(f"  training_log: {os.path.basename(path)}")


def import_embeddings(cur, prefix, npz_path):
    data = np.load(npz_path)
    emb, model = data["emb"], str(data["model"])
    cur.execute("SELECT n FROM tsmini.dataset WHERE name = %s", (prefix,))
    n = cur.fetchone()[0]
    if len(emb) != n:
        print(f"  {npz_path} has {len(emb)} rows but dataset {prefix} has {n}, skipping embeddings")
        return
    if exists(cur, "SELECT 1 FROM trajcl.embedding WHERE dataset = %s AND model = %s", prefix, model):
        cur.execute("DELETE FROM trajcl.embedding WHERE dataset = %s AND model = %s", (prefix, model))
    with cur.copy("COPY trajcl.embedding (dataset, model, idx, embedding) FROM STDIN WITH (FORMAT binary)") as copy:
        copy.set_types(["text", "text", "int4", "float4[]"])
        for i, e in enumerate(emb):
            copy.write_row((prefix, model, i, e.tolist()))
    print(f"  embedding: {emb.shape[0]} x {emb.shape[1]} (model {model})")


def restore(cur, root, prefix, force):
    cur.execute("SELECT path, size FROM trajcl.file WHERE path LIKE %s ORDER BY path",
                ("%/" + prefix.replace("_", "\\_") + "\\_%",))
    rows = cur.fetchall()
    if not rows:
        print(f"nothing saved for {prefix}")
    for rel, size in rows:
        path = os.path.join(root, *rel.split("/"))
        if os.path.exists(path) and not force:
            print(f"  {rel} already on disk (use --force to overwrite)")
            continue
        cur.execute("SELECT content FROM trajcl.file WHERE path = %s", (rel,))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(cur.fetchone()[0])
        print(f"  restored {rel} ({size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("BEIJING_DATA_ROOT", HERE), help="TrajCL bundle folder")
    ap.add_argument("--data-root", default=os.environ.get("BEIJING_DATA_ROOT", ""), help="optional folder containing bundled TrajCL/data/<prefix>* inputs")
    ap.add_argument("--prefix", default="beijing_20200", help="dataset prefix, e.g. rome_20200 or beijing_20200")
    ap.add_argument("--restore", action="store_true", help="write the saved files back to disk instead")
    ap.add_argument("--force", action="store_true", help="with --restore: overwrite files that already exist")
    args = ap.parse_args()

    with connect() as conn, conn.cursor() as cur:
        if args.restore:
            if exists(cur, "SELECT 1 FROM pg_tables WHERE schemaname = 'trajcl' AND tablename = 'file'"):
                restore(cur, args.root, args.prefix, args.force)
            else:
                print("nothing saved yet (no trajcl schema)")
            print("done")
            return

        has_dataset = ensure_tsmini_dataset(cur, args.prefix)
        with open(os.path.join(SQL_DIR, "04_trajcl.sql")) as f:
            cur.execute(f.read())
        print("files:")
        import_files(cur, args.root, args.prefix)
        if args.data_root:
            import_test_inputs(cur, args.data_root, args.prefix)
        import_logs(cur, args.root, args.prefix)
        npz = os.path.join(args.root, "exp", f"embeddings_{args.prefix}.npz")
        if not os.path.isfile(npz):
            print(f"  {npz} not found, skipping embeddings (run embed.py after fine-tuning)")
        elif has_dataset:
            import_embeddings(cur, args.prefix, npz)
        print("committing...")
    print("done")


if __name__ == "__main__":
    main()
