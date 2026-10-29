#!/usr/bin/env python3
"""
Embed trajectories with a trained TSMini encoder and search them.

Run from inside the TSMini folder (venv active). Needs prepare_rome.py next to it
(for the exact Frechet function) and the patched config.py (dataset 'rome').

  python search_rome.py embed                       # embed every trajectory, cache to exp/
  python search_rome.py query --idx 5 --k 10        # most similar trajectories to #5
  python search_rome.py query --idx 5 --plot q.png  # ... and draw them on a map
  python search_rome.py eval  --nq 200              # how well does embedding search match exact search?

How it works: the encoder maps each trajectory to a 128-number vector; TSMini was
trained so that the L1 distance between two vectors approximates the Frechet distance
between the trajectories. So "most similar trajectories" = nearest vectors.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from numba import njit, prange

from config import Config
from utils.cellspace import create_cellspace
from utils.traj import preprocess_traj, padding_traj
from model.tsmini import TSMini
from prepare_rome import frechet  # exact discrete Frechet (numba)


# ----------------------------------------------------------------------------- data
def load_all(h5_path):
    with h5py.File(h5_path, "r") as f:
        n = int(f.attrs["n"])
        merc_range = f.attrs["merc_range"][()].tolist()
        trajs = [f[f"/trajs_merc/{i}"][:] for i in range(n)]
    return trajs, merc_range


def pad_all(trajs):
    P = np.zeros((len(trajs), max(map(len, trajs)), 2))
    L = np.zeros(len(trajs), dtype=np.int64)
    for i, t in enumerate(trajs):
        P[i, : len(t)] = t
        L[i] = len(t)
    return P, L


@njit(parallel=True, cache=True)
def exact_one_to_all(q, lq, P, L):
    out = np.empty(P.shape[0])
    for i in prange(P.shape[0]):
        out[i] = frechet(q, lq, P[i], L[i])
    return out


# ----------------------------------------------------------------------------- model
def checkpoint_path():
    return f"{Config.checkpoint_dir}/{Config.dataset_prefix}_trajsimi_{Config.trajsimi_encoder_name}_" \
           f"{Config.trajsimi_measure_fn_name}_best{Config.dumpfile_uniqueid}.pt"


def load_model(ckpt):
    model = TSMini().to(Config.device)
    model.load_state_dict(torch.load(ckpt, map_location=Config.device)["encoder"])
    model.eval()
    return model


@torch.no_grad()
def embed(model, trajs, space, bs=256):
    outs = []
    for s in range(0, len(trajs), bs):
        batch = [preprocess_traj(t.tolist(), space, 0) for t in trajs[s : s + bs]]
        x, l = padding_traj(batch)
        x = torch.tensor(x, dtype=torch.float).to(Config.device)
        l = torch.tensor(l, dtype=torch.long).to(Config.device)
        outs.append(model(x, l).cpu())
        if (s // bs) % 20 == 0:
            print(f"  embedded {min(s + bs, len(trajs))}/{len(trajs)}", flush=True)
    return torch.cat(outs).numpy()


def get_embeddings(trajs, space, force=False):
    ckpt = checkpoint_path()
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    cache = f"{Config.root_dir}/exp/embeddings_{Config.dataset_prefix}.npz"
    stamp = os.path.getmtime(ckpt)
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        if float(z["stamp"]) == stamp and len(z["emb"]) == len(trajs):
            print(f"loaded cached embeddings {z['emb'].shape} from {cache}")
            return z["emb"]
    print(f"embedding {len(trajs)} trajectories on {Config.device} ...")
    t0 = time.time()
    E = embed(load_model(ckpt), trajs, space)
    print(f"done in {time.time() - t0:.1f}s")
    np.savez(cache, emb=E, stamp=stamp)
    print(f"saved {cache}")
    return E


def l1_to_all(E, q):
    d = np.abs(E - E[q]).sum(axis=1)
    d[q] = np.inf  # a trajectory is not its own neighbour
    return d


def split_of(idx, n):
    off = int(n * 0.7)
    if off <= idx < off + 7000:
        return "train (seen in training)"
    if off + 7000 <= idx < off + 8000:
        return "eval (used for early stopping)"
    if off + 8000 <= idx < off + 10000:
        return "test (held out)"
    return "unused in training"


# ----------------------------------------------------------------------------- modes
def mode_query(args, trajs, E, P, L):
    n = len(trajs)
    q = args.idx
    d_emb = l1_to_all(E, q)
    top_emb = np.argsort(d_emb)[: args.k]
    d_ex = exact_one_to_all(P[q], L[q], P, L)
    d_ex[q] = np.inf
    top_ex = np.argsort(d_ex)[: args.k]
    cand = np.argsort(d_emb)[: args.rerank]
    top_rr = cand[np.argsort(d_ex[cand])][: args.k]

    print(f"\nquery #{q}: {len(trajs[q])} points, {split_of(q, n)}")
    print(f"\n{'rank':>4} {'idx':>7} {'emb L1':>9} {'exact Frechet (m)':>18}  in exact top-{args.k}?")
    exact_set = set(top_ex.tolist())
    for r, i in enumerate(top_emb, 1):
        print(f"{r:>4} {i:>7} {d_emb[i]:>9.3f} {d_ex[i]:>18.1f}  {'yes' if i in exact_set else 'no'}")
    hit = len(exact_set & set(top_emb.tolist()))
    hit_rr = len(exact_set & set(top_rr.tolist()))
    print(f"\nembedding search found {hit}/{args.k} of the true top-{args.k}")
    print(f"embedding top-{args.rerank} + exact rerank found {hit_rr}/{args.k}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed: uv pip install matplotlib")
            return
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in top_emb:
            ax.plot(trajs[i][:, 0], trajs[i][:, 1], color="tab:blue", alpha=0.5, lw=1)
        ax.plot(trajs[q][:, 0], trajs[q][:, 1], color="tab:red", lw=2.5, label=f"query #{q}")
        ax.plot([], [], color="tab:blue", label=f"top-{args.k} by embedding")
        ax.set_aspect("equal"); ax.legend(); ax.set_title("TSMini most-similar search (Web Mercator, m)")
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"saved {args.plot}")


def mode_eval(args, trajs, E, P, L):
    n = len(trajs)
    off = int(n * 0.7)
    pool = np.array([i for i in range(n) if not (off <= i < off + 7000)])  # not in the training slice
    rng = np.random.default_rng(args.seed)
    qs = rng.choice(pool, size=min(args.nq, len(pool)), replace=False)
    hr10, r10_50, t_emb, t_ex = [], [], 0.0, 0.0
    for q in qs:
        t0 = time.time()
        d_emb = l1_to_all(E, q)
        order = np.argsort(d_emb)
        t_emb += time.time() - t0
        t0 = time.time()
        d_ex = exact_one_to_all(P[q], L[q], P, L)
        d_ex[q] = np.inf
        ex10 = set(np.argsort(d_ex)[:10].tolist())
        t_ex += time.time() - t0
        hr10.append(len(ex10 & set(order[:10].tolist())) / 10)
        r10_50.append(len(ex10 & set(order[:50].tolist())) / 10)
    print(f"\n{len(qs)} random queries not in the training slice, database = all {n} trajectories")
    print(f"HR@10   (embedding top-10 vs exact top-10): {np.mean(hr10):.3f}")
    print(f"R10@50  (exact top-10 inside embedding top-50): {np.mean(r10_50):.3f}")
    print(f"time per query: embedding {1000 * t_emb / len(qs):.2f} ms, exact Frechet {1000 * t_ex / len(qs):.1f} ms "
          f"(exact uses all CPU cores; embeddings are already computed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["embed", "query", "eval"])
    ap.add_argument("--dataset", default="rome")
    ap.add_argument("--measure", default="frechet")
    ap.add_argument("--idx", type=int, default=0, help="query trajectory index (query mode)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank", type=int, default=50, help="candidates re-ranked with exact Frechet")
    ap.add_argument("--plot", default="", help="save a map of the query and its neighbours to this png")
    ap.add_argument("--nq", type=int, default=200, help="number of queries (eval mode)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reembed", action="store_true")
    args = ap.parse_args()

    Config.update({"dataset": args.dataset, "trajsimi_measure_fn_name": args.measure})
    trajs, merc_range = load_all(Config.dataset_file + ".h5")
    space = create_cellspace(*merc_range, Config.cell_size, Config.cell_size, Config.cellspace_buffer)
    E = get_embeddings(trajs, space, force=args.reembed)
    if args.mode == "embed":
        return
    if args.mode == "query" and not (0 <= args.idx < len(trajs)):
        sys.exit(f"--idx must be in 0..{len(trajs) - 1}")
    P, L = pad_all(trajs)
    (mode_query if args.mode == "query" else mode_eval)(args, trajs, E, P, L)


if __name__ == "__main__":
    main()
