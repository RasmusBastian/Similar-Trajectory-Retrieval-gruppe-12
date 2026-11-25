#!/usr/bin/env python3
"""
Compare TSMini and TrajCL on the same dataset, the same queries and the same metrics.

Both models embed every trajectory; a search returns the nearest embeddings (L1). This script checks those
searches against exact Frechet search for the same random queries, and collects the test-set numbers each
model's training log reports. Needs numpy, pandas, h5py and matplotlib; TrajCL/.venv has them:

    TrajCL/.venv/bin/python scripts/compare_models.py --prefix beijing_20200 --nq 100
    TrajCL/.venv/bin/python scripts/compare_models.py --prefix beijing_20200 --query 5   # also a side-by-side map

Reads   TrajCL/data/<prefix>                                trajectories (same order as TSMini/data/<prefix>.h5)
        TSMini/exp/embeddings_<prefix>.npz                  TSMini/search_rome.py embed --dataset <city>
        TrajCL/exp/embeddings_<prefix>.npz                  TrajCL/embed.py
        TSMini/exp/log/*.log, TrajCL/exp/log/*.log          the [EXPFlag] line of each model's training run
Writes  results/<prefix>/comparison.json, comparison.png, query_<idx>.png
        results/<prefix>/exact_cache.npz                    exact Frechet per query (git-ignored), so a re-run
                                                            with a newly trained model takes seconds

A model whose embeddings are missing is skipped, so this also runs with one model.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'TrajCL'))
from search_trajcl import exact_one_to_all, l1_to_all, pad_all, plt_or_exit  # noqa: E402

MODELS = {'TSMini': 'TSMini', 'TrajCL': 'TrajCL'}   # model name -> folder
COLORS = {'TSMini': 'tab:orange', 'TrajCL': 'tab:blue'}
KS = (10, 50, 100, 200, 500, 1000)


def load_trajs(prefix):
    import pandas as pd
    path = os.path.join(ROOT, 'TrajCL', 'data', prefix)
    if not os.path.isfile(path):
        sys.exit('missing {} (unpack the team zip with scripts/share_data.py)'.format(path))
    return [np.asarray(t, dtype=np.float64) for t in pd.read_pickle(path).merc_seq.values]


def load_embeddings(prefix, n, overrides):
    embs = {}
    for name, folder in MODELS.items():
        path = overrides.get(name) or os.path.join(ROOT, folder, 'exp', 'embeddings_{}.npz'.format(prefix))
        if not os.path.isfile(path):
            print('{}: no embeddings at {}, skipped'.format(name, os.path.relpath(path, ROOT)))
            continue
        E = np.load(path)['emb']
        if len(E) != n:
            sys.exit('{}: {} embeddings but {} trajectories'.format(name, len(E), n))
        embs[name] = E
        print('{}: {} x {} embeddings from {}'.format(name, *E.shape, os.path.relpath(path, ROOT)))
    if not embs:
        sys.exit('no embeddings found for {}'.format(prefix))
    return embs


def test_metrics(name, prefix):
    """hr5/hr20/hr20in5 on the 2,000 test trajectories, from the newest training log with an [EXPFlag] line."""
    pat = re.compile(r'\[EXPFlag\]model={},dataset={},.*'.format(name, prefix))
    for path in sorted(glob.glob(os.path.join(ROOT, MODELS[name], 'exp', 'log', '*.log')), reverse=True):
        with open(path, encoding='utf-8', errors='replace') as f:
            m = pat.search(f.read())
        if m:
            vals = dict(re.findall(r'(hr\w+)=([\d.]+)', m.group(0)))
            out = {k: float(vals[k]) for k in ('hr5', 'hr20', 'hr20in5') if k in vals}
            out['log'] = os.path.relpath(path, ROOT).replace(os.sep, '/')
            return out
    return {}


def exact_distances(prefix, trajs, qs, out_dir):
    """Exact Frechet from each query to every trajectory, cached per query in exact_cache.npz."""
    cache = os.path.join(out_dir, 'exact_cache.npz')
    known = {}
    if os.path.isfile(cache):
        z = np.load(cache)
        known = dict(zip(z['qs'].tolist(), z['D']))
    todo = [q for q in qs if q not in known]
    if todo:
        print('exact Frechet for {} queries ({} cached), about {:.0f} s each ...'.format(len(todo), len(qs) - len(todo), 12))
        P, L = pad_all(trajs)
        for c, q in enumerate(todo, 1):
            known[q] = exact_one_to_all(trajs[q], P, L).astype(np.float32)
            if c % 10 == 0 or c == len(todo):
                print('  {}/{}'.format(c, len(todo)), flush=True)
                np.savez(cache, qs=np.array(list(known)), D=np.stack(list(known.values())))
    else:
        print('exact Frechet: all {} queries cached'.format(len(qs)))
    D = np.stack([known[q] for q in qs]).astype(np.float64)
    D[np.arange(len(qs)), qs] = np.inf   # a trajectory is not its own neighbour
    return D


def evaluate(E, qs, D):
    """For every query: where the exact top-10 land in the embedding ranking, and the search time."""
    n = len(E)
    ranks, t = [], 0.0
    for row, q in enumerate(qs):
        t0 = time.time()
        order = np.argsort(l1_to_all(E, q))
        t += time.time() - t0
        pos = np.empty(n, dtype=np.int64)
        pos[order] = np.arange(1, n + 1)
        ranks.append(np.sort(pos[np.argsort(D[row])[:10]]))
    ranks = np.array(ranks)
    return {
        'hr10': float((ranks <= 10).mean()),
        'recall': {str(k): float((ranks <= k).mean()) for k in KS},
        'ms_per_query': 1000 * t / len(qs),
        '_ranks': ranks,
    }


def plot_comparison(res, prefix, nq, n, path):
    plt = plt_or_exit()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ks = np.unique(np.geomspace(10, 1000, 60).astype(int))
    ax = axes[0]
    for name, r in res.items():
        ax.plot(ks, [(r['_ranks'] <= k).mean() for k in ks], lw=2, color=COLORS[name], label=name)
    ax.set_xscale('log'); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc='lower right')
    ax.set_xlabel('shortlist size k (embedding search), log scale')
    ax.set_ylabel('share of exact top-10 found')
    ax.set_title('Shortlist needed before exact re-ranking')

    ax = axes[1]
    bars = [('HR@10', lambda r: r['hr10']), ('R10@50', lambda r: r['recall']['50']),
            ('R10@200', lambda r: r['recall']['200']), ('test HR@5', lambda r: r['test'].get('hr5', np.nan)),
            ('test R5@20', lambda r: r['test'].get('hr20in5', np.nan))]
    width = 0.8 / len(res)
    for j, (name, r) in enumerate(res.items()):
        vals = [f(r) for _, f in bars]
        xs = np.arange(len(bars)) + (j - (len(res) - 1) / 2) * width
        ax.bar(xs, vals, width, color=COLORS[name], label=name)
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.01, '{:.2f}'.format(v), ha='center', fontsize=8)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b for b, _ in bars])
    ax.set_ylim(0, 1.08); ax.legend(loc='upper left')
    ax.set_title('Search quality (left 3: {} queries vs all {}; right 2: training test set)'.format(nq, n), fontsize=10)
    fig.suptitle('TSMini vs TrajCL, {}'.format(prefix), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved {}'.format(os.path.relpath(path, ROOT)))


def plot_query(trajs, embs, q, D_q, prefix, path, k=10):
    plt = plt_or_exit()
    exact = set(np.argsort(D_q)[:k].tolist())
    tops = {name: np.argsort(l1_to_all(E, q))[:k] for name, E in embs.items()}
    origin = trajs[q][0]
    rel = lambda t: (t - origin) / 1000.0
    pts = np.vstack([rel(trajs[i]) for i in [q, *np.concatenate(list(tops.values()))]])
    (x0, y0), (x1, y1) = pts.min(axis=0) - 0.2, pts.max(axis=0) + 0.2
    fig, axes = plt.subplots(1, len(tops), figsize=(6 * len(tops), 7), squeeze=False)
    for ax, (name, top) in zip(axes[0], tops.items()):
        for i in top:
            ls = '-' if i in exact else ':'
            ax.plot(*rel(trajs[i]).T, color=COLORS[name], alpha=0.6, lw=1.2, ls=ls)
        ax.plot(*rel(trajs[q]).T, color='red', lw=2.5)
        hits = len(exact & set(top.tolist()))
        ax.set_title('{}: {}/{} of the exact top-{}'.format(name, hits, k, k))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal')
        ax.set_xlabel('km east'); ax.set_ylabel('km north')
    fig.legend(handles=[plt.Line2D([], [], color='red', lw=2.5, label='query #{}'.format(q)),
                        plt.Line2D([], [], color='grey', lw=1.2, label='match in the exact top-{}'.format(k)),
                        plt.Line2D([], [], color='grey', lw=1.2, ls=':', label='match not in the exact top-{}'.format(k))],
               loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Query #{}: top-{} by embedding, {}'.format(q, k, prefix), fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('saved {}'.format(os.path.relpath(path, ROOT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='beijing_20200')
    ap.add_argument('--nq', type=int, default=100, help='number of random queries')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--query', type=int, default=None, help='also draw this trajectory with both models\' top-10')
    ap.add_argument('--out', default=None, help='output folder (default results/<prefix>)')
    ap.add_argument('--tsmini-emb', default=None, help='use this npz instead of TSMini/exp/embeddings_<prefix>.npz')
    ap.add_argument('--trajcl-emb', default=None, help='use this npz instead of TrajCL/exp/embeddings_<prefix>.npz')
    args = ap.parse_args()

    out_dir = args.out or os.path.join(ROOT, 'results', args.prefix)
    os.makedirs(out_dir, exist_ok=True)
    trajs = load_trajs(args.prefix)
    n = len(trajs)
    embs = load_embeddings(args.prefix, n, {'TSMini': args.tsmini_emb, 'TrajCL': args.trajcl_emb})

    # same queries as TrajCL/search_trajcl.py eval with the same seed: not in the fine-tuning train slice
    off = int(n * 0.7)
    pool = np.setdiff1d(np.arange(n), np.arange(off, off + 7000))
    qs = np.random.default_rng(args.seed).choice(pool, size=args.nq, replace=False)
    extra = [args.query] if args.query is not None and args.query not in qs else []
    D_all = exact_distances(args.prefix, trajs, np.concatenate([qs, extra]).astype(int), out_dir)
    D = D_all[:len(qs)]

    res = {}
    for name, E in embs.items():
        res[name] = evaluate(E, qs, D)
        res[name]['test'] = test_metrics(name, args.prefix)

    print('\n{} queries, database = all {} trajectories, exact Frechet as reference'.format(args.nq, n))
    print('{:<8} {:>7} {}  {:>9}   {}'.format('model', 'HR@10', ' '.join('R10@{:<5}'.format(k) for k in KS[1:]),
                                             'ms/query', 'test HR@5 / HR@20 / R5@20'))
    for name, r in res.items():
        t = r['test']
        print('{:<8} {:>7.3f} {}  {:>9.1f}   {}'.format(
            name, r['hr10'], ' '.join('{:<10.3f}'.format(r['recall'][str(k)]) for k in KS[1:]), r['ms_per_query'],
            '{:.3f} / {:.3f} / {:.3f}'.format(t['hr5'], t['hr20'], t['hr20in5']) if 'hr5' in t else 'no log'))

    summary = {'prefix': args.prefix, 'n': n, 'nq': args.nq, 'seed': args.seed, 'queries': qs.tolist(),
               'models': {name: {k: v for k, v in r.items() if not k.startswith('_')} for name, r in res.items()}}
    with open(os.path.join(out_dir, 'comparison.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('saved {}'.format(os.path.relpath(os.path.join(out_dir, 'comparison.json'), ROOT)))
    plot_comparison(res, args.prefix, args.nq, n, os.path.join(out_dir, 'comparison.png'))
    if args.query is not None:
        plot_query(trajs, embs, args.query, D_all[-1] if extra else D[list(qs).index(args.query)],
                   args.prefix, os.path.join(out_dir, 'query_{}.png'.format(args.query)))


if __name__ == '__main__':
    main()
