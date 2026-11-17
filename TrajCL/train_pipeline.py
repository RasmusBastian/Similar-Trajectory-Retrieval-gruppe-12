#!/usr/bin/env python3
"""
Run TrajCL's whole training pipeline for one dataset, skipping steps that are already done.

    .venv-gpu/bin/python train_pipeline.py --dataset beijing             # run everything still missing
    .venv-gpu/bin/python train_pipeline.py --dataset beijing --save-db   # also save results in the project DB
    .venv-gpu/bin/python train_pipeline.py --dataset beijing --only pretrain
    .venv-gpu/bin/python train_pipeline.py --dataset beijing --from finetune

Steps, in order (each one needs the output of the one before it):

    prepare    dataset files from ../TSMini/data (trajectories, Frechet ground truth, test sets)
    node2vec   cell embeddings: a vector per 100 m grid cell, learned from where trajectories go
    pretrain   contrastive pre-training of the TrajCL encoder (train.py)
    finetune   fine-tuning to approximate the Frechet distance (train_trajsimi.py)
    embed      one embedding per trajectory, for similarity search (embed.py)

A step counts as done when its output file exists. To redo a step, delete its output
(paths are printed when the script starts) or use --only / --from, which always run.

Prerequisite: ../TSMini/data/<prefix>.h5 and <prefix>_traj_simi_dict_frechet.pkl
(made by ../TSMini/prepare_rome.py, see the root README).
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

from config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURE = 'frechet'   # config.py defaults to EDwP, which we don't have ground truth for
STEPS = ['prepare', 'node2vec', 'pretrain', 'finetune', 'embed']


def outputs(dataset):
    """The file each step produces. Its existence means the step is done."""
    Config.update({'dataset': dataset})
    p = Config.dataset_prefix
    snap = Config.checkpoint_dir
    return {
        'prepare':  Config.dataset_file + '_newsimi_raw.pkl',        # written last by the prepare step
        'node2vec': Config.dataset_embs_file,
        'pretrain': '{}/{}_TrajCL_best.pt'.format(snap, p),
        'finetune': '{}/{}_trajsimi_TrajCL_{}_best.pt'.format(snap, p, MEASURE),
        'embed':    '{}/exp/embeddings_{}.npz'.format(HERE, p),
    }


def commands(dataset):
    """The command each step runs. sys.executable = the Python running this script (.venv-gpu)."""
    py = sys.executable
    prep = 'prepare_{}.py'.format(dataset)
    return {
        'prepare':  [py, prep],
        'node2vec': [py, prep, '--node2vec'],   # also redoes 'prepare', which takes under a minute
        'pretrain': [py, 'train.py', '--dataset', dataset],
        'finetune': [py, 'train_trajsimi.py', '--dataset', dataset, '--trajsimi_measure_fn_name', MEASURE],
        'embed':    [py, 'embed.py', '--dataset', dataset, '--trajsimi_measure_fn_name', MEASURE],
    }


def run(cmd, log_path):
    """Run a command, showing its output and also writing it to log_path. Stops the script if it fails."""
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    if proc.returncode != 0:
        sys.exit('\nFAILED (exit code {}): {}\nfull output: {}'.format(proc.returncode, ' '.join(cmd), log_path))


def save_to_db(dataset):
    """Store embeddings, models and logs in the project DB (schema trajcl), so nobody has to retrain."""
    Config.update({'dataset': dataset})
    root = os.path.dirname(HERE)
    cmd = ['docker', 'compose', 'run', '--rm', '--build', 'trajcl-import', '--prefix', Config.dataset_prefix]
    print('\n=== saving to project database: {}'.format(' '.join(cmd)), flush=True)
    if subprocess.run(cmd, cwd=root).returncode != 0:
        if database_mode():
            sys.exit('saving required TrajCL inputs to the database failed; later stages cannot load them')
        print('WARNING: saving to the database failed. The files on disk are fine; run this again later:\n'
              '  cd {} && {}'.format(root, ' '.join(cmd)))


def database_mode():
    return (Config.dataset == 'beijing'
            or os.environ.get('TRAINING_DATA_SOURCE', '').lower() in ('db', 'database')
            or bool(os.environ.get('PGHOST')))


def check_database_inputs(dataset_prefix):
    from utils.data_loader import _db_connection
    with _db_connection() as conn, conn.cursor() as cur:
        cur.execute('''SELECT EXISTS (SELECT 1 FROM tsmini.dataset WHERE name = %s),
                              EXISTS (SELECT 1 FROM tsmini.ground_truth_set WHERE dataset = %s AND measure = %s)''',
                    (dataset_prefix, dataset_prefix, MEASURE))
        has_dataset, has_truth = cur.fetchone()
    if not has_dataset or not has_truth:
        missing = 'trajectory dataset' if not has_dataset else 'Fréchet ground truth'
        sys.exit(f'{missing} for {dataset_prefix} is missing from the database; import the dataset zip first')


def database_step_exists(step, prefix):
    """Use PostgreSQL as the cache index; never restore saved inputs to local files."""
    from utils.data_loader import _db_connection
    with _db_connection() as conn, conn.cursor() as cur:
        if step == 'prepare':
            cur.execute("SELECT count(DISTINCT (variant, rate)) = 11 FROM trajcl.test_trajectory WHERE dataset = %s", (prefix,))
        elif step == 'node2vec':
            cur.execute("""SELECT count(*) = 2 FROM trajcl.file WHERE path IN (%s, %s)""",
                        (f'data/{prefix}_cell100_cellspace.pkl', f'data/{prefix}_cell100_embdim256_embs.pkl'))
        elif step == 'pretrain':
            cur.execute("SELECT EXISTS (SELECT 1 FROM trajcl.file WHERE path = %s)",
                        (f'exp/snapshots/{prefix}_TrajCL_best.pt',))
        elif step == 'finetune':
            cur.execute("SELECT EXISTS (SELECT 1 FROM trajcl.file WHERE path = %s)",
                        (f'exp/snapshots/{prefix}_trajsimi_TrajCL_frechet_best.pt',))
        elif step == 'embed':
            cur.execute("""SELECT EXISTS (
                           SELECT 1 FROM trajcl.embedding e JOIN tsmini.dataset d ON d.name = e.dataset
                           WHERE e.dataset = %s AND e.model = %s AND
                                 (SELECT count(*) FROM trajcl.embedding x
                                  WHERE x.dataset = e.dataset AND x.model = e.model) = d.n)""",
                        (prefix, f'{prefix}_trajsimi_TrajCL_frechet_best.pt'))
        return cur.fetchone()[0]


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return '{}h{:02d}m{:02d}s'.format(h, m, s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['beijing', 'rome'])
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--only', choices=STEPS, help='run just this step, even if it is done')
    group.add_argument('--from', dest='start', choices=STEPS, help='run this step and all later ones')
    ap.add_argument('--save-db', action='store_true',
                    help='after node2vec, finetune and embed, save results in the project DB (needs its db running)')
    args = ap.parse_args()

    Config.update({'dataset': args.dataset})
    tsmini = os.path.join(HERE, '..', 'TSMini', 'data', Config.dataset_prefix)
    if database_mode():
        check_database_inputs(Config.dataset_prefix)
    else:
        for f in [tsmini + '.h5', tsmini + '_traj_simi_dict_{}.pkl'.format(MEASURE)]:
            if not os.path.isfile(f):
                sys.exit('missing input: {}\nimport the Beijing zip or set database mode (see the root README)'.format(f))

    import torch
    print('device: {}'.format(torch.cuda.get_device_name(0) if torch.cuda.is_available()
                                else 'CPU (no GPU found: this will be slow)'))

    outs, cmds = outputs(args.dataset), commands(args.dataset)
    if args.only:
        forced = [args.only]
    elif args.start:
        forced = STEPS[STEPS.index(args.start):]
    else:
        forced = []
    todo = [s for s in STEPS if s in forced or (not forced and not (
        database_step_exists(s, Config.dataset_prefix) if database_mode() else os.path.isfile(outs[s]))) ]

    print('\nsteps for {}:'.format(Config.dataset_prefix))
    for s in STEPS:
        print('  {:9} {:5}  {}'.format(s, 'RUN' if s in todo else 'skip', outs[s]))
    if not todo:
        print('\nnothing to do: every step is done. Use --only or --from to redo a step.')
        return

    os.makedirs(os.path.join(HERE, 'exp', 'log'), exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timings = []
    t_all = time.time()
    for s in todo:
        log_path = os.path.join(HERE, 'exp', 'log', 'pipeline_{}_{}_{}.log'.format(Config.dataset_prefix, stamp, s))
        print('\n=== {}  started {}  ({})'.format(s, datetime.now().strftime('%H:%M:%S'), ' '.join(cmds[s][1:])),
              flush=True)
        t0 = time.time()
        run(cmds[s], log_path)
        if not os.path.isfile(outs[s]):
            sys.exit('step {} finished but its output is missing: {}'.format(s, outs[s]))
        timings.append((s, time.time() - t0))
        print('=== {}  done in {}'.format(s, fmt(timings[-1][1])), flush=True)
        if database_mode() or (args.save_db and s in ('node2vec', 'pretrain', 'finetune', 'embed')):
            save_to_db(args.dataset)

    print('\nall done in {}'.format(fmt(time.time() - t_all)))
    for s, t in timings:
        print('  {:9} {}'.format(s, fmt(t)))


if __name__ == '__main__':
    main()
