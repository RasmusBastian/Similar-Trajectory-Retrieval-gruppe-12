#!/usr/bin/env python3
"""
Checks that prepare.py -> train.py -> embed.py run on the installed dependencies, using synthetic
data in a temp folder. It says nothing about model quality. Run from this folder:

    .venv/bin/python smoke_test.py

The "ground truth" is the distance between trajectory centroids, not Frechet, to keep it fast.
TSMini's layout needs at least 33,334 trajectories, so the synthetic set has that many (short ones).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import embed
import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(prepare.N_TOTAL / 0.3) + 1


def synthetic(rng):
    """Random walks of 20-40 points in a ~5 km box in Web-Mercator metres (roughly Rome)."""
    x0, y0 = 1.3775e6, 5.1305e6
    trajs = []
    for _ in range(N):
        n = rng.integers(20, 41)
        start = np.array([x0, y0]) + rng.uniform(0, 5000, 2)
        trajs.append(start + np.cumsum(rng.normal(0, 60, (n, 2)), axis=0))
    return trajs


def fake_simi(trajs):
    off = int(len(trajs) * 0.7)
    cents = np.array([t.mean(axis=0) for t in trajs[off:off + prepare.N_TOTAL]])
    mats = []
    for a, b in ((0, 7000), (7000, 8000), (8000, 10000)):
        c = cents[a:b]
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        mats.append(np.triu(d, 1))
    return mats[0], mats[1], mats[2], float(max(m.max() for m in mats))


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix='convtraj_smoke_')
    try:
        prefix = 'smoke'
        data_dir, exp_dir = os.path.join(tmp, 'data'), os.path.join(tmp, 'exp')
        trajs = synthetic(rng)
        prepare.build(trajs, fake_simi(trajs), data_dir, prefix)

        # coordinates round-trip through lon/lat
        back = prepare.merc_to_lonlat(trajs[0])
        x = np.radians(back[:, 0]) * prepare.R_EARTH
        assert np.allclose(x, trajs[0][:, 0], atol=1e-3)

        cmd = [sys.executable, 'train.py', '--train_flag', embed.train_flag(prefix), '--dist_type', 'frechet',
               '--epoch_num', '1', '--test_epoch', '1', '--root_read_path', data_dir, '--root_write_path', exp_dir]
        subprocess.check_call(cmd, cwd=HERE)

        npz = os.path.join(exp_dir, 'embeddings.npz')
        metrics = embed.run(data_dir, exp_dir, prefix, npz)
        emb = np.load(npz)['emb']
        assert emb.shape == (N, 128) and np.isfinite(emb).all()
        assert all(0 <= v <= 1 for v in metrics.values())
        print('OK smoke test:', json.dumps({k: round(v, 3) for k, v in metrics.items()}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
