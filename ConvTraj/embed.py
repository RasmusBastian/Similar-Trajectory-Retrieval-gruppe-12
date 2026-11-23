#!/usr/bin/env python3
"""
After train.py: put ConvTraj's embeddings back in h5 order and evaluate them on TSMini's
test split with the same metrics as TSMini and TrajCL (HR@5, HR@20, R5@20).

    .venv/bin/python embed.py --dataset beijing

Reads   exp/<prefix>/feature_dir/frechet_feature_<prefix>_frechet   embeddings in traj_list order (train.py)
        data/<prefix>/meta.json, data/<prefix>/frechet_test_truth.npy (prepare.py)
Writes  exp/embeddings_<prefix>.npz   emb: [n, 128] float32, row i = trajectory i (= tsmini.trajectory.idx),
                                      model: checkpoint file name
        exp/<prefix>/metrics.json

The Euclidean (L2) distance between two embeddings approximates Frechet / max_distance,
because that is the distance ConvTraj's triplet loss trains (all_feature_distance_type = euclidean).
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIXES = {'rome': 'rome_20200', 'beijing': 'beijing_20200'}
MEASURE = 'frechet'


def train_flag(prefix):
    return '{}_{}'.format(prefix, MEASURE)


def hitting_ratio(preds, truths, pred_topk, truth_topk):
    """Same as TrajSimi.hitting_ratio in TSMini/TrajCL: top-k overlap, the query itself excluded."""
    _, p = torch.topk(preds, pred_topk + 1, dim=1, largest=False)
    _, t = torch.topk(truths, truth_topk + 1, dim=1, largest=False)
    tp = sum(np.intersect1d(p[i].numpy(), t[i].numpy()).size for i in range(p.shape[0]))
    return (tp - preds.shape[0]) / (truth_topk * preds.shape[0])


def evaluate(test_emb, truth):
    pred = torch.cdist(torch.as_tensor(test_emb), torch.as_tensor(test_emb), p=2)
    truth = torch.as_tensor(truth)
    return {'hr5': hitting_ratio(pred, truth, 5, 5),
            'hr20': hitting_ratio(pred, truth, 20, 20),
            'hr20in5': hitting_ratio(pred, truth, 20, 5)}


def run(data_dir, exp_dir, prefix, out_npz):
    meta = json.load(open(os.path.join(data_dir, 'meta.json')))
    flag = train_flag(prefix)
    feature_path = os.path.join(exp_dir, 'feature_dir', '{}_feature_{}'.format(MEASURE, flag))
    if not os.path.isfile(feature_path):
        raise SystemExit('embeddings not found: {} (run train.py first)'.format(feature_path))
    emb_listorder = np.asarray(pickle.load(open(feature_path, 'rb')), dtype=np.float32)

    order = np.asarray(meta['order'])
    assert len(emb_listorder) == len(order)
    emb = np.empty_like(emb_listorder)
    emb[order] = emb_listorder

    q0, q1 = meta['train_set'], meta['train_set'] + meta['query_set']
    metrics = evaluate(emb_listorder[q0:q1], np.load(os.path.join(data_dir, '{}_test_truth.npy'.format(MEASURE))))
    print('test ({} trajectories): '.format(q1 - q0) + ', '.join('{} {:.4f}'.format(k, v) for k, v in metrics.items()))
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    np.savez(out_npz, emb=emb, model=flag)
    print('wrote {} ({} x {})'.format(out_npz, *emb.shape))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(PREFIXES))
    args = ap.parse_args()
    prefix = PREFIXES[args.dataset]
    run(os.path.join(HERE, 'data', prefix), os.path.join(HERE, 'exp', prefix), prefix,
        os.path.join(HERE, 'exp', 'embeddings_{}.npz'.format(prefix)))


if __name__ == '__main__':
    main()
