#!/usr/bin/env python3
"""
Train TSMini on 20, 40, 60, 80 and 100% of the training trajectories and collect the test scores.

    .venv-gpu/bin/python run_fraction_experiment.py                      # all five fractions
    .venv-gpu/bin/python run_fraction_experiment.py --fractions 0.2 0.4  # only some

Every run uses the same 1,000 eval and 2,000 test trajectories and the same number of batches per
epoch, so the only thing that changes is how many training trajectories the model learns from.
Each run saves its model as ..._best_frac20.pt etc., so the normal model (..._best.pt) is not touched.

Results are appended to exp/fraction_experiment_<prefix>.csv. Fractions already in that file are
skipped, so the script can be restarted after an interruption.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

from config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ['fraction', 'n_train', 'hr5', 'hr20', 'hr20in5', 'hr10', 'hr50', 'hr50in10',
          'best_epoch', 'best_hr_eval', 'train_time_s', 'log']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='beijing')
    ap.add_argument('--measure', default='frechet')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--fractions', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8, 1.0])
    args = ap.parse_args()

    Config.update({'dataset': args.dataset})
    prefix = Config.dataset_prefix
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

        log_path = os.path.join(HERE, 'exp', 'log', 'fraction_{}_{}pct.log'.format(prefix, pct))
        cmd = [sys.executable, 'train_trajsimi.py',
               '--dataset', args.dataset,
               '--trajsimi_measure_fn_name', args.measure,
               '--trajsimi_epoch', str(args.epochs),
               '--trajsimi_train_frac', str(frac),
               '--dumpfile_uniqueid', '_frac{}'.format(pct)]
        print('\n=== {}% of the training trajectories  ({})'.format(pct, time.strftime('%H:%M:%S')), flush=True)

        lines = []
        with open(log_path, 'w') as log:
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                lines.append(line)
            proc.wait()
        if proc.returncode != 0:
            sys.exit('\n{}% FAILED (exit code {}), see {}'.format(pct, proc.returncode, log_path))

        text = ''.join(lines)
        flag = re.search(r'\[EXPFlag\].*', text)
        end = re.search(r'training end\. @=([\d.]+), best_epoch=(\d+), best_hr_eval=([\d.]+)', text)
        used = re.search(r'using (\d+) of (\d+) training trajectories', text)
        test = re.search(r'traj: #total=\d+ \(trains/evals/tests=(\d+)/', text)
        if not flag or not end:
            sys.exit('\n{}%: could not find the results in the output, see {}'.format(pct, log_path))
        scores = dict(re.findall(r'(hr\w+)=([\d.]+)', flag.group(0)))

        row = {'fraction': frac,
               'n_train': used.group(1) if used else (test.group(1) if test else ''),
               'best_epoch': end.group(2), 'best_hr_eval': end.group(3),
               'train_time_s': round(float(end.group(1))), 'log': os.path.relpath(log_path, HERE)}
        for k in ('hr5', 'hr20', 'hr20in5', 'hr10', 'hr50', 'hr50in10'):
            row[k] = scores.get(k, '')

        new_file = not os.path.isfile(out)
        with open(out, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        print('=== {}% done: test HR@5={} HR@20={} R5@20={}  (best epoch {}, {} s)'.format(
            pct, row['hr5'], row['hr20'], row['hr20in5'], row['best_epoch'], row['train_time_s']), flush=True)

    print('\nresults: {}'.format(out))
    with open(out) as fh:
        print(fh.read())


if __name__ == '__main__':
    main()
