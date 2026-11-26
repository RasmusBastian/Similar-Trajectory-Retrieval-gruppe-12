#!/usr/bin/env python3
"""
Train ConvTraj on 20, 40, 60, 80 and 100% of the 7,000 training trajectories and collect the test scores,
the same experiment as TSMini/run_fraction_experiment.py and TrajCL/run_fraction_experiment.py.

    .venv/bin/python run_fraction_experiment.py --dataset beijing
    .venv/bin/python run_fraction_experiment.py --dataset beijing --fractions 0.2 0.4

Needs data/<prefix>/ from prepare.py (train_pipeline.py runs it). Every run:
  - uses upstream's --train_ratio, which keeps the first share of the training trajectories. prepare.py keeps
    TSMini's order, so these are the same trajectories as in the TSMini and TrajCL runs, and the subsets are nested;
  - has as many anchors (and batches) per epoch as the full training set (local change in mynn/Traj_KNN.py);
  - trains 200 epochs and keeps the last one, as upstream does, with the same seed (666);
  - is scored on the same 2,000 test trajectories with embed.py's hitting_ratio.
Each run writes to exp/<prefix>_frac<pct>/, so the normal model in exp/<prefix>/ is not touched.
Results are appended to exp/fraction_experiment_<prefix>.csv; fractions already there are skipped.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

from embed import MEASURE, PREFIXES, run as embed_run, train_flag

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ['fraction', 'n_train', 'hr5', 'hr20', 'hr20in5', 'epochs', 'train_time_s', 'log']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=list(PREFIXES))
    ap.add_argument('--fractions', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    prefix = PREFIXES[args.dataset]
    data_dir = os.path.join(HERE, 'data', prefix)
    if not os.path.isfile(os.path.join(data_dir, 'meta.json')):
        sys.exit('missing {}/meta.json: run prepare.py (or train_pipeline.py) first'.format(data_dir))
    n_full = json.load(open(os.path.join(data_dir, 'meta.json')))['train_set']

    os.makedirs(os.path.join(HERE, 'exp', 'log'), exist_ok=True)
    out = os.path.join(HERE, 'exp', 'fraction_experiment_{}.csv'.format(prefix))
    done = set()
    if os.path.isfile(out):
        with open(out) as fh:
            done = {round(float(r['fraction']), 4) for r in csv.DictReader(fh)}

    for frac in args.fractions:
        pct = int(round(frac * 100))
        if round(frac, 4) in done:
            print('{}%: already in {}, skipping'.format(pct, out))
            continue
        exp_dir = os.path.join(HERE, 'exp', '{}_frac{}'.format(prefix, pct))
        log_path = os.path.join(HERE, 'exp', 'log', 'fraction_{}_{}pct.log'.format(prefix, pct))
        cmd = [sys.executable, '-u', 'train.py', '--train_flag', train_flag(prefix), '--dist_type', MEASURE,
               '--epoch_num', str(args.epochs), '--test_epoch', str(args.epochs), '--device', args.device,
               '--train_ratio', str(frac), '--root_read_path', data_dir, '--root_write_path', exp_dir]
        print('\n=== {}% of the training trajectories  ({})'.format(pct, time.strftime('%H:%M:%S')), flush=True)

        t0 = time.time()
        with open(log_path, 'w') as log:
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
        train_time = time.time() - t0
        if proc.returncode != 0:
            sys.exit('\n{}% FAILED (exit code {}), see {}'.format(pct, proc.returncode, log_path))

        metrics = embed_run(data_dir, exp_dir, prefix, os.path.join(exp_dir, 'embeddings.npz'))
        row = {'fraction': frac, 'n_train': int(n_full * frac), 'epochs': args.epochs,
               'train_time_s': round(train_time), 'log': os.path.relpath(log_path, HERE)}
        row.update({k: round(float(v), 6) for k, v in metrics.items()})

        new_file = not os.path.isfile(out)
        with open(out, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        print('=== {}% done: test HR@5={} HR@20={} R5@20={}  ({} s)'.format(
            pct, row['hr5'], row['hr20'], row['hr20in5'], row['train_time_s']), flush=True)

    print('\nresults: {}'.format(out))
    with open(out) as fh:
        print(fh.read())


if __name__ == '__main__':
    main()
