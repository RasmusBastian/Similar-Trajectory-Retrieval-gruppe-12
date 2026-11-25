#!/usr/bin/env python3
"""
Search trajectories with TrajCL's embeddings and plot the results, like ../TSMini/search_rome.py does for TSMini.

Needs no model or GPU: it reads the embeddings embed.py wrote (exp/embeddings_<prefix>.npz) and the trajectories
in data/<prefix>. Plots need matplotlib (uv pip install matplotlib). Run from this folder:

  python search_trajcl.py query --idx 5 --k 10                    # most similar trajectories to #5
  python search_trajcl.py query --idx 5 --plot q.png --grid g.png # ... drawn together, and one panel per match
  python search_trajcl.py eval --nq 100 --plot eval.png           # how well does embedding search match exact search?
  python search_trajcl.py curves --plot curves.png                # training curves from exp/log/pipeline_*.log

The fine-tuned model maps each trajectory to 256 numbers, trained so that the L1 distance between two
embeddings approximates the Frechet distance between the trajectories. "Most similar" = nearest embeddings.
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- data
def load(prefix):
    emb_path = os.path.join(HERE, 'exp', 'embeddings_{}.npz'.format(prefix))
    data_path = os.path.join(HERE, 'data', prefix)
    for p in (emb_path, data_path):
        if not os.path.isfile(p):
            sys.exit('missing {} (run embed.py, or unpack the team zip with scripts/share_data.py)'.format(p))
    z = np.load(emb_path)
    trajs = [np.asarray(t, dtype=np.float64) for t in pd.read_pickle(data_path).merc_seq.values]
    if len(trajs) != len(z['emb']):
        sys.exit('{} trajectories but {} embeddings'.format(len(trajs), len(z['emb'])))
    return trajs, z['emb'], str(z['model'])


def pad_all(trajs):
    P = np.zeros((len(trajs), max(map(len, trajs)), 2))
    L = np.array([len(t) for t in trajs])
    for i, t in enumerate(trajs):
        P[i, :len(t)] = t
    return P, L


def exact_one_to_all(q, P, L):
    """Discrete Frechet distance from q to every trajectory, one dynamic-programming pass over all of them at once
    (the same recurrence as TSMini/prepare_rome.py's frechet). Padding columns are never read for the result."""
    n, m = P.shape[0], P.shape[1]
    prev = np.empty((n, m))
    cur = np.empty((n, m))
    d = np.linalg.norm(P - q[0], axis=2)                 # (n, m): distance of q[0] to every point
    prev[:, 0] = d[:, 0]
    for j in range(1, m):
        prev[:, j] = np.maximum(prev[:, j - 1], d[:, j])
    for i in range(1, len(q)):
        d = np.linalg.norm(P - q[i], axis=2)
        cur[:, 0] = np.maximum(prev[:, 0], d[:, 0])
        best = np.minimum(prev[:, 1:], prev[:, :-1])     # min(prev[j], prev[j-1]) for every j >= 1
        for j in range(1, m):
            cur[:, j] = np.maximum(np.minimum(best[:, j - 1], cur[:, j - 1]), d[:, j])
        prev, cur = cur, prev
    return prev[np.arange(n), L - 1]


def l1_to_all(E, q):
    d = np.abs(E - E[q]).sum(axis=1)
    d[q] = np.inf  # a trajectory is not its own neighbour
    return d


def split_of(idx, n):
    """Fine-tuning uses 10,000 trajectories from 70% into the dataset (utils/data_loader.py), like TSMini."""
    off = int(n * 0.7)
    if off <= idx < off + 7000:
        return 'train (seen in fine-tuning)'
    if off + 7000 <= idx < off + 8000:
        return 'eval (used to pick the best epoch)'
    if off + 8000 <= idx < off + 10000:
        return 'test (held out)'
    return 'not used in fine-tuning'


def plt_or_exit():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        sys.exit('matplotlib not installed: uv pip install matplotlib')


# ----------------------------------------------------------------------------- modes
def mode_query(args):
    trajs, E, model = load(args.prefix)
    n, q = len(trajs), args.idx
    if not 0 <= q < n:
        sys.exit('--idx must be in 0..{}'.format(n - 1))
    P, L = pad_all(trajs)
    d_emb = l1_to_all(E, q)
    top_emb = np.argsort(d_emb)[:args.k]
    t0 = time.time()
    d_ex = exact_one_to_all(trajs[q], P, L)
    d_ex[q] = np.inf
    t_ex = time.time() - t0
    top_ex = np.argsort(d_ex)[:args.k]
    cand = np.argsort(d_emb)[:args.rerank]
    top_rr = cand[np.argsort(d_ex[cand])][:args.k]

    print('model {}, {} trajectories'.format(model, n))
    print('\nquery #{}: {} points, {}'.format(q, len(trajs[q]), split_of(q, n)))
    print('\n{:>4} {:>7} {:>9} {:>18}  in exact top-{}?'.format('rank', 'idx', 'emb L1', 'exact Frechet (m)', args.k))
    exact_set = set(top_ex.tolist())
    for r, i in enumerate(top_emb, 1):
        print('{:>4} {:>7} {:>9.3f} {:>18.1f}  {}'.format(r, i, d_emb[i], d_ex[i], 'yes' if i in exact_set else 'no'))
    print('\nembedding search found {}/{} of the true top-{}'.format(len(exact_set & set(top_emb.tolist())), args.k, args.k))
    print('embedding top-{} + exact rerank found {}/{}'.format(args.rerank, len(exact_set & set(top_rr.tolist())), args.k))
    print('exact Frechet against all {} took {:.1f}s'.format(n, t_ex))

    if args.plot:
        plt = plt_or_exit()
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in top_emb:
            ax.plot(trajs[i][:, 0], trajs[i][:, 1], color='tab:blue', alpha=0.5, lw=1)
        ax.plot(trajs[q][:, 0], trajs[q][:, 1], color='tab:red', lw=2.5, label='query #{}'.format(q))
        ax.plot([], [], color='tab:blue', label='top-{} by embedding'.format(args.k))
        ax.set_aspect('equal'); ax.legend()
        ax.set_title('TrajCL most-similar search, {} (Web Mercator, m)'.format(args.prefix))
        fig.savefig(args.plot, dpi=150, bbox_inches='tight'); plt.close(fig)
        print('saved {}'.format(args.plot))

    if args.grid:
        plt = plt_or_exit()
        # km from the query's start, the same window in every panel so they can be compared
        origin = trajs[q][0]
        rel = lambda t: (t - origin) / 1000.0
        pts = np.vstack([rel(trajs[i]) for i in [q, *top_emb]])
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        pad = 0.05 * max(x1 - x0, y1 - y0)
        aspect = (x1 - x0 + 2 * pad) / (y1 - y0 + 2 * pad)       # width / height of the window
        ncols = 5 if aspect < 1 else 2
        nrows = (len(top_emb) + ncols - 1) // ncols
        h = 4.2 if aspect < 1 else 3.6
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * max(2.6, h * aspect) + 0.5, nrows * (h + 0.9)),
                                 squeeze=False, sharex=True, sharey=True)
        for rank, i in enumerate(top_emb, 1):
            ax = axes[(rank - 1) // ncols][(rank - 1) % ncols]
            ax.plot(*rel(trajs[q]).T, color='red', lw=2)
            ax.plot(*rel(trajs[i]).T, color='tab:blue', lw=1.4)
            ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad); ax.set_aspect('equal')
            ax.set_title('Rank {}: #{}{}\nFrechet {:.0f} m | L1 {:.3f}'.format(
                rank, i, '' if i in exact_set else ' *', d_ex[i], d_emb[i]), fontsize=10)
        for ax in axes[-1]:
            ax.set_xlabel('km east')
        for ax in axes[:, 0]:
            ax.set_ylabel('km north')
        for ax in axes.flat[len(top_emb):]:
            ax.remove()
        fig.legend(handles=[plt.Line2D([], [], color='red', lw=2, label='query #{}'.format(q)),
                            plt.Line2D([], [], color='tab:blue', lw=1.4, label='match from TrajCL')],
                   loc='lower center', ncol=2, frameon=False)
        fig.suptitle("Query #{} compared with TrajCL's top {} ({})\n* = not in the exact Frechet top-{}".format(
            q, len(top_emb), args.prefix, args.k), fontsize=13)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(args.grid, dpi=150, bbox_inches='tight'); plt.close(fig)
        print('saved {}'.format(args.grid))


def mode_eval(args):
    trajs, E, model = load(args.prefix)
    n = len(trajs)
    P, L = pad_all(trajs)
    off = int(n * 0.7)
    pool = np.setdiff1d(np.arange(n), np.arange(off, off + 7000))  # not seen in fine-tuning
    qs = np.random.default_rng(args.seed).choice(pool, size=args.nq, replace=False)
    hr10, r10_50, t_emb, t_ex = [], [], 0.0, 0.0
    ranks = []  # per query: where the exact top-10 land in the embedding ranking (1 = first)
    for c, q in enumerate(qs, 1):
        t0 = time.time()
        d_emb = l1_to_all(E, q)
        order = np.argsort(d_emb)
        top_emb = order[:50]
        t_emb += time.time() - t0
        t0 = time.time()
        d_ex = exact_one_to_all(trajs[q], P, L)
        d_ex[q] = np.inf
        t_ex += time.time() - t0
        top_ex = set(np.argsort(d_ex)[:10].tolist())
        hr10.append(len(top_ex & set(top_emb[:10].tolist())) / 10)
        r10_50.append(len(top_ex & set(top_emb.tolist())) / 10)
        pos = np.empty(n, dtype=np.int64)
        pos[order] = np.arange(1, n + 1)
        ranks.append(np.sort(pos[list(top_ex)]))
        if c % 10 == 0:
            print('  {}/{} queries, HR@10 so far {:.3f}'.format(c, args.nq, np.mean(hr10)), flush=True)
    print('\nmodel {}'.format(model))
    print('{} random queries not seen in fine-tuning, database = all {} trajectories'.format(args.nq, n))
    print('HR@10   (embedding top-10 vs exact top-10): {:.3f}'.format(np.mean(hr10)))
    print('R10@50  (exact top-10 inside embedding top-50): {:.3f}'.format(np.mean(r10_50)))
    print('time per query: embedding {:.2f} ms, exact Frechet {:.0f} ms'.format(
        1000 * t_emb / args.nq, 1000 * t_ex / args.nq))

    ranks = np.array(ranks)                                  # (nq, 10)
    ks = np.unique(np.geomspace(10, 1000, 60).astype(int))
    recall = [(ranks <= k).mean() for k in ks]               # share of the exact top-10 inside the top-k shortlist
    for k in (10, 50, 100, 200, 500, 1000):
        print('R10@{:<5} {:.3f}   (shortlist {} = {:.1f}% of the database)'.format(
            k, (ranks <= k).mean(), k, 100 * k / n))
    if args.plot:
        plt = plt_or_exit()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        ax = axes[0]
        ax.plot(ks, recall, color='tab:blue', lw=2)
        for k in (10, 50, 200):
            v = (ranks <= k).mean()
            ax.plot([k], [v], 'o', color='tab:blue')
            ax.annotate('top-{}: {:.2f}'.format(k, v), (k, v), textcoords='offset points', xytext=(8, -14))
        ax.set_xscale('log'); ax.set_ylim(0, 1)
        ax.set_xlabel('shortlist size k (embedding search), log scale')
        ax.set_ylabel('share of exact top-10 found')
        ax.set_title('How big a shortlist is needed before exact re-ranking?')
        ax.grid(alpha=0.3)
        ax = axes[1]
        hits = np.round(np.array(hr10) * 10).astype(int)
        ax.bar(range(11), np.bincount(hits, minlength=11), color='tab:blue')
        ax.axvline(hits.mean(), color='tab:red', ls='--', label='mean {:.1f} (HR@10 {:.3f})'.format(hits.mean(), np.mean(hr10)))
        ax.set_xticks(range(11)); ax.legend()
        ax.set_xlabel('true top-10 neighbours found in embedding top-10'); ax.set_ylabel('number of queries')
        ax.set_title('Per query: embedding top-10 vs exact top-10')
        fig.suptitle('TrajCL search quality, {} ({} queries, database {} trajectories)'.format(args.prefix, args.nq, n),
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(args.plot, dpi=150, bbox_inches='tight'); plt.close(fig)
        print('saved {}'.format(args.plot))


def latest_log(prefix, step):
    logs = sorted(glob.glob(os.path.join(HERE, 'exp', 'log', 'pipeline_{}_*_{}.log'.format(prefix, step))))
    return open(logs[-1], encoding='utf-8', errors='replace').read() if logs else ''


def mode_curves(args):
    n2v = [(int(e), float(l)) for e, l in re.findall(r'\[node2vec\] i_ep=(\d+), loss=([\d.]+)', latest_log(args.prefix, 'node2vec'))]
    pre = [(int(e), float(l)) for e, l in re.findall(r'\[Training\] ep=(\d+): avg_loss=([\d.]+)', latest_log(args.prefix, 'pretrain'))]
    ft_log = latest_log(args.prefix, 'finetune')
    ft_train = [(int(e), float(l)) for e, l in re.findall(r'training\. i_ep=(\d+), loss=([\d.]+)', ft_log)]
    ft_eval = [(int(r[0]), *map(float, r[1:])) for r in
               re.findall(r'eval\.\s+i_ep=(\d+), loss=([\d.]+), hr=([\d.]+),([\d.]+),([\d.]+)', ft_log)]
    test = re.search(r'hr5=([\d.]+),hr20=([\d.]+),hr20in5=([\d.]+)', ft_log)
    if not (n2v or pre or ft_eval):
        sys.exit('no pipeline logs for {} in exp/log'.format(args.prefix))

    print('node2vec  : {} epochs, final loss {}'.format(len(n2v), n2v[-1][1] if n2v else '-'))
    print('pretrain  : {} epochs, best loss {}'.format(len(pre), min(l for _, l in pre) if pre else '-'))
    if ft_eval:
        best = max(ft_eval, key=lambda r: (r[2], r[0]))  # the last of tied epochs, like train_trajsimi.py
        print('finetune  : {} epochs, best eval HR@5 {:.3f} at epoch {}'.format(len(ft_eval), best[2], best[0]))
    if test:
        print('test set  : HR@5 {} | HR@20 {} | R5@20 {}'.format(*test.groups()))

    plt = plt_or_exit()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    if n2v:
        ax.plot(*zip(*n2v), marker='o', ms=3)
    ax.set_title('1. node2vec cell embeddings'); ax.set_xlabel('epoch'); ax.set_ylabel('training loss')
    ax = axes[1]
    if pre:
        ax.plot(*zip(*pre), marker='o', ms=3, color='tab:purple')
    ax.set_title('2. contrastive pre-training'); ax.set_xlabel('epoch'); ax.set_ylabel('average loss')
    ax = axes[2]
    if ft_eval:
        ep = [r[0] for r in ft_eval]
        for col, name, color in ((2, 'HR@5', 'tab:blue'), (3, 'HR@20', 'tab:orange'), (4, 'R5@20', 'tab:green')):
            ax.plot(ep, [r[col] for r in ft_eval], marker='o', ms=3, label='eval ' + name, color=color)
        if test:
            for v, color in zip(map(float, test.groups()), ('tab:blue', 'tab:orange', 'tab:green')):
                ax.axhline(v, color=color, ls='--', lw=1)
            ax.plot([], [], color='grey', ls='--', label='test set (best model)')
        ax.set_ylim(0, 1); ax.legend(loc='lower right')
    ax.set_title('3. fine-tuning to Frechet'); ax.set_xlabel('epoch'); ax.set_ylabel('hit ratio')
    fig.suptitle('TrajCL training, {}'.format(args.prefix), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.plot or 'curves_{}.png'.format(args.prefix)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved {}'.format(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['query', 'eval', 'curves'])
    ap.add_argument('--prefix', default='beijing_20200', help='dataset, e.g. beijing_20200 or rome_20200')
    ap.add_argument('--idx', type=int, default=0, help='query trajectory index (query mode)')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--rerank', type=int, default=50, help='candidates re-ranked with exact Frechet')
    ap.add_argument('--plot', default='', help='query: all matches on one map; eval: search-quality graphs; '
                                               'curves: output png')
    ap.add_argument('--grid', default='', help='query: one panel per match, query in red')
    ap.add_argument('--nq', type=int, default=100, help='number of queries (eval mode)')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    {'query': mode_query, 'eval': mode_eval, 'curves': mode_curves}[args.mode](args)


if __name__ == '__main__':
    main()
