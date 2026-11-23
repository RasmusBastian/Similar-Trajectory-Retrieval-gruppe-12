#!/usr/bin/env python3
"""
Run ConvTraj's whole pipeline for one dataset, skipping steps that are already done.

    .venv/bin/python train_pipeline.py --dataset beijing
    .venv/bin/python train_pipeline.py --dataset beijing --from train --epochs 50

Steps, in order (each one needs the output of the one before it):

    prepare   dataset files from ../TSMini/data (prepare.py)
    train     train ConvTraj on Frechet, then embed every trajectory (upstream train.py)
    embed     reorder embeddings to h5 order, evaluate HR@5/HR@20/R5@20 on the test split (embed.py)

A step counts as done when its output file exists; --from always reruns from that step.
"""
import argparse
import os
import subprocess
import sys

from embed import MEASURE, PREFIXES, train_flag

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['prepare', 'train', 'embed']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(PREFIXES))
    ap.add_argument('--from', dest='start', choices=STEPS)
    ap.add_argument('--epochs', type=int, default=200, help='upstream used 200 in its experiments')
    ap.add_argument('--device', default='cuda:0', help='falls back to mps, then cpu')
    args = ap.parse_args()

    prefix = PREFIXES[args.dataset]
    data_dir, exp_dir = 'data/' + prefix, 'exp/' + prefix
    flag = train_flag(prefix)
    py = sys.executable
    steps = {
        'prepare': ('{}/meta.json'.format(data_dir), [py, 'prepare.py', '--dataset', args.dataset]),
        'train': ('{}/feature_dir/{}_feature_{}'.format(exp_dir, MEASURE, flag),
                  [py, 'train.py', '--train_flag', flag, '--dist_type', MEASURE,
                   '--epoch_num', str(args.epochs), '--test_epoch', str(args.epochs), '--device', args.device,
                   '--root_read_path', data_dir, '--root_write_path', exp_dir]),
        'embed': ('exp/embeddings_{}.npz'.format(prefix), [py, 'embed.py', '--dataset', args.dataset]),
    }
    forced = STEPS[STEPS.index(args.start):] if args.start else []
    for name in STEPS:
        out, cmd = steps[name]
        if name not in forced and os.path.isfile(os.path.join(HERE, out)):
            print('[{}] done, skipping ({})'.format(name, out))
            continue
        print('[{}] {}'.format(name, ' '.join(cmd)), flush=True)
        if subprocess.call(cmd, cwd=HERE) != 0:
            sys.exit('step {} failed'.format(name))
        forced = STEPS[STEPS.index(name) + 1:]  # later steps depend on this output, so redo them


if __name__ == '__main__':
    main()
