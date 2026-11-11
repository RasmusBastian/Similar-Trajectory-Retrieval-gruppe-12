#!/usr/bin/env python3
"""
Share TSMini/TrajCL data and trained models with the team as one zip per dataset.
Only needs the Python standard library. Run from anywhere; paths are relative to the repository root.

    python scripts/share_data.py pack --prefix beijing_20200    # -> shared/beijing_20200_<date>_<name>.zip
    python scripts/share_data.py list shared/<file>.zip          # who packed it and what is inside
    python scripts/share_data.py unpack shared/<file>.zip        # put the files in place (add --force, see below)

Workflow: pack -> upload the zip to the team's Google Drive folder -> teammates download and unpack ->
train -> pack again -> upload. Unpack never overwrites a local file that differs from the one in the zip
(e.g. your own newer model) unless you pass --force, so agree who trains which model.

Packed (only what exists), for prefix P:
    TSMini/data/P.h5, TSMini/data/P_traj_simi_dict_*.pkl           trajectories and ground truth
    TSMini/exp/snapshots/P_*.pt, TSMini/exp/embeddings_P.npz        trained model, embeddings
    TrajCL/data/P, P_newsimi_*.pkl, P_cell*_cellspace.pkl, P_cell*_embs.pkl   dataset, node2vec cell embeddings
    TrajCL/exp/snapshots/P_*.pt, TrajCL/exp/embeddings_P.npz        node2vec, pre-trained, fine-tuned models
    TSMini/exp/log/*.log, TrajCL/exp/log/*.log                      training logs
TrajCL/data/P_traj_simi_dict_*.pkl is a copy of TSMini's, so it is not packed but copied by unpack.
Files tracked by git (e.g. TSMini's committed Rome model) are skipped: teammates get them with git pull.
"""
import argparse
import datetime
import getpass
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = "manifest.json"


def bundle_patterns(p):
    return [
        f"TSMini/data/{p}.h5",
        f"TSMini/data/{p}_traj_simi_dict_*.pkl",
        f"TSMini/exp/snapshots/{p}_*.pt",
        f"TSMini/exp/embeddings_{p}.npz",
        "TSMini/exp/log/*.log",
        f"TrajCL/data/{p}",
        f"TrajCL/data/{p}_newsimi_*.pkl",
        f"TrajCL/data/{p}_cell*_cellspace.pkl",
        f"TrajCL/data/{p}_cell*_embs.pkl",
        f"TrajCL/exp/snapshots/{p}_*.pt",
        f"TrajCL/exp/embeddings_{p}.npz",
        "TrajCL/exp/log/*.log",
    ]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def mb(n):
    return f"{n / 1e6:.1f} MB"


def local(rel):
    return os.path.join(ROOT, *rel.split("/"))


# ----------------------------------------------------------------------------- pack
def pack(args):
    # files tracked by git reach teammates through git already (and git may change their line endings)
    tracked = set(git("ls-files", "TSMini", "TrajCL").splitlines())
    rels = sorted({os.path.relpath(path, ROOT).replace(os.sep, "/")
                   for pattern in bundle_patterns(args.prefix)
                   for path in glob.glob(local(pattern)) if os.path.isfile(path)} - tracked)
    if not any(r.startswith(f"TSMini/data/{args.prefix}") for r in rels):
        sys.exit(f"TSMini/data/{args.prefix}.h5 not found, nothing to share for prefix {args.prefix}")

    who = git("config", "user.name") or getpass.getuser()
    now = datetime.datetime.now()
    files = {}
    for rel in rels:
        files[rel] = {"size": os.path.getsize(local(rel)), "sha256": sha256(local(rel))}
    manifest = {"prefix": args.prefix, "packed_by": who, "packed_at": now.isoformat(timespec="seconds"),
                "git_commit": git("rev-parse", "--short", "HEAD"), "files": files}

    out = args.out or os.path.join(
        ROOT, "shared", f"{args.prefix}_{now:%Y%m%d-%H%M}_{re.sub(r'[^A-Za-z0-9]+', '-', who).strip('-')}.zip")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2))
        for rel in rels:
            print(f"  {rel} ({mb(files[rel]['size'])})", flush=True)
            z.write(local(rel), rel)
    os.replace(tmp, out)
    print(f"wrote {out} ({mb(os.path.getsize(out))}, {len(rels)} files)")
    print("Upload it to the team's Google Drive folder.")


# ----------------------------------------------------------------------------- list
def read_manifest(z):
    try:
        return json.loads(z.read(MANIFEST))
    except KeyError:
        sys.exit(f"{z.filename} has no {MANIFEST}, not made by share_data.py pack")


def show(args):
    with zipfile.ZipFile(args.zip) as z:
        m = read_manifest(z)
    print(f"prefix {m['prefix']}, packed by {m['packed_by']} at {m['packed_at']} (commit {m['git_commit'] or '?'})")
    for rel, info in m["files"].items():
        print(f"  {rel} ({mb(info['size'])})")


# ----------------------------------------------------------------------------- unpack
def unpack(args):
    with zipfile.ZipFile(args.zip) as z:
        m = read_manifest(z)
        print(f"prefix {m['prefix']}, packed by {m['packed_by']} at {m['packed_at']}")

        new, same, conflicts = [], [], []
        for rel, info in m["files"].items():
            if rel.startswith("/") or ".." in rel.split("/"):
                sys.exit(f"refusing unsafe path in zip: {rel}")
            path = local(rel)
            if not os.path.exists(path):
                new.append(rel)
            elif os.path.getsize(path) == info["size"] and sha256(path) == info["sha256"]:
                same.append(rel)
            else:
                conflicts.append(rel)

        if conflicts and not args.force:
            print("These local files differ from the zip and were NOT overwritten (nothing was written):")
            for rel in conflicts:
                print(f"  {rel}")
            sys.exit("Keep yours, or rerun with --force to replace them with the zip's versions.")

        for rel in new + conflicts:
            info = m["files"][rel]
            path = local(rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
            h = hashlib.sha256()
            try:
                with z.open(rel) as src, os.fdopen(fd, "wb") as dst:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        h.update(chunk)
                        dst.write(chunk)
                if h.hexdigest() != info["sha256"]:
                    sys.exit(f"checksum mismatch for {rel}, the zip is damaged (download it again)")
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            print(f"  {'replaced' if rel in conflicts else 'added'} {rel} ({mb(info['size'])})", flush=True)

    # TrajCL uses the same ground truth as TSMini; prepare_*.py copies it, so do the same here
    p = m["prefix"]
    if os.path.exists(local(f"TrajCL/data/{p}")):
        for src in glob.glob(local(f"TSMini/data/{p}_traj_simi_dict_*.pkl")):
            dst = local("TrajCL/data/" + os.path.basename(src))
            if not os.path.exists(dst) or sha256(dst) != sha256(src):
                shutil.copyfile(src, dst)
                print(f"  copied {os.path.basename(src)} to TrajCL/data")

    print(f"{len(new)} added, {len(conflicts)} replaced, {len(same)} already identical")
    print("Load it into your local database (from the repository root, project db running):")
    print(f"  docker compose run --rm --build tsmini-import --prefix {p}")
    print(f"  docker compose run --rm --build trajcl-import --prefix {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pack", help="zip a dataset's data, models and logs")
    a.add_argument("--prefix", default="beijing_20200", help="e.g. beijing_20200 or rome_20200")
    a.add_argument("--out", help="zip path (default: shared/<prefix>_<date>_<name>.zip)")
    a.set_defaults(fn=pack)
    a = sub.add_parser("list", help="show who packed a zip and what is inside")
    a.add_argument("zip")
    a.set_defaults(fn=show)
    a = sub.add_parser("unpack", help="put a zip's files in place")
    a.add_argument("zip")
    a.add_argument("--force", action="store_true", help="replace local files that differ from the zip")
    a.set_defaults(fn=unpack)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
