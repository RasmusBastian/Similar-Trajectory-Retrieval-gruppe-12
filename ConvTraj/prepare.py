#!/usr/bin/env python3
"""
Build ConvTraj's input files from TSMini's, so all three models use the same trajectories,
splits and Frechet ground truth. Run from this folder after TSMini has its files
(TSMini/prepare_rome.py, see the root README):

    .venv/bin/python prepare.py --dataset rome
    .venv/bin/python prepare.py --dataset beijing

Reads   ../TSMini/data/<prefix>.h5                              trajectories (Web-Mercator metres)
        ../TSMini/data/<prefix>_traj_simi_dict_frechet.pkl      ground truth (upper-triangular, metres)
Writes  data/<prefix>/traj_list                                 list of [[lon, lat], ...], in ConvTraj's order
        data/<prefix>/frechet_train_distance_matrix_result      7000 x 7000 float32, Frechet / max_distance
        data/<prefix>/frechet_test_truth.npy                    2000 x 2000 float32, test split, for embed.py
        data/<prefix>/meta.json                                 split sizes and the row order (see below)

ConvTraj expects one list ordered train | query | base. TSMini uses 10,000 trajectories from
index int(0.7 * n): 7000 train, 1000 eval, 2000 test. Here:
    train = TSMini's train split (7000), the only part ConvTraj trains on
    query = TSMini's test split (2000), evaluated by embed.py exactly like TSMini/TrajCL do
    base  = every other trajectory (incl. TSMini's eval split), embedded only for similarity search
meta.json's "order"[k] is the h5 index (= tsmini.trajectory.idx) of row k of traj_list.

Distances are divided by max_distance like TSMini does, so they are 0..1 and the triplet loss
works on the same scale as upstream's Porto/Geolife distances (which are in degrees).
"""
import argparse
import json
import math
import os
import pickle

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TSMINI_DATA = os.path.join(HERE, '..', 'TSMini', 'data')
PREFIXES = {'rome': 'rome_20200', 'beijing': 'beijing_20200'}
R_EARTH = 6378137.0  # Web Mercator sphere radius, as in TSMini/prepare_rome.py
N_TOTAL, N_TRAIN, N_EVAL, N_TEST = 10000, 7000, 1000, 2000
MEASURE = 'frechet'


def merc_to_lonlat(xy):
    xy = np.asarray(xy, dtype=np.float64)
    lon = np.degrees(xy[:, 0] / R_EARTH)
    lat = np.degrees(2 * np.arctan(np.exp(xy[:, 1] / R_EARTH)) - math.pi / 2)
    return np.stack([lon, lat], axis=1)


def full_matrix(upper, max_distance):
    """TSMini stores the upper triangle in metres; ConvTraj wants the full symmetric matrix."""
    m = np.asarray(upper, dtype=np.float64)
    return ((m + m.T) / max_distance).astype(np.float32)


def split_indices(n):
    off = int(n * 0.7)
    train = list(range(off, off + N_TRAIN))
    test = list(range(off + N_TRAIN + N_EVAL, off + N_TOTAL))
    taken = set(train) | set(test)
    rest = [i for i in range(n) if i not in taken]
    return train, test, rest


def build(trajs_merc, simi, out_dir, prefix):
    """trajs_merc: list of (len, 2) arrays in h5 order. simi: TSMini's pkl tuple."""
    trains_simi, _evals_simi, tests_simi, max_distance = simi
    n = len(trajs_merc)
    assert n >= int(N_TOTAL / 0.3) + 1, 'TSMini reads 10000 trajectories from int(0.7*n), got n={}'.format(n)
    train, test, rest = split_indices(n)
    order = train + test + rest

    traj_list = [merc_to_lonlat(trajs_merc[i]).tolist() for i in order]
    train_matrix = full_matrix(trains_simi, max_distance)
    assert train_matrix.shape == (N_TRAIN, N_TRAIN)
    assert np.asarray(tests_simi).shape == (N_TEST, N_TEST)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'traj_list'), 'wb') as f:
        pickle.dump(traj_list, f, protocol=4)
    with open(os.path.join(out_dir, '{}_train_distance_matrix_result'.format(MEASURE)), 'wb') as f:
        pickle.dump(train_matrix, f, protocol=4)
    # not named *_test_distance_matrix_result on purpose: train.py would load it as query x base
    np.save(os.path.join(out_dir, '{}_test_truth.npy'.format(MEASURE)), full_matrix(tests_simi, max_distance))
    meta = {'prefix': prefix, 'measure': MEASURE, 'max_distance': float(max_distance),
            'train_set': len(train), 'query_set': len(test), 'base_set': len(rest), 'order': order}
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print('wrote {}: {} trajectories (train {}, query {}, base {}), max_distance {:.1f} m'.format(
        out_dir, n, len(train), len(test), len(rest), max_distance))


def load_tsmini(prefix):
    h5_path = os.path.join(TSMINI_DATA, prefix + '.h5')
    pkl_path = os.path.join(TSMINI_DATA, '{}_traj_simi_dict_{}.pkl'.format(prefix, MEASURE))
    for p in (h5_path, pkl_path):
        if not os.path.isfile(p):
            raise SystemExit('missing {} (run TSMini/prepare_rome.py or unpack the shared zip first)'.format(p))
    with h5py.File(h5_path, 'r') as f:
        n = int(f.attrs['n'])
        trajs = [f['/trajs_merc/{}'.format(i)][:] for i in range(n)]
    with open(pkl_path, 'rb') as f:
        simi = pickle.load(f)
    return trajs, simi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(PREFIXES))
    args = ap.parse_args()
    prefix = PREFIXES[args.dataset]
    trajs, simi = load_tsmini(prefix)
    build(trajs, simi, os.path.join(HERE, 'data', prefix), prefix)


if __name__ == '__main__':
    main()
