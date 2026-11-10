#!/usr/bin/env python3
"""
Build TrajCL's Rome dataset from TSMini's, so both models use the same trajectories,
splits and ground truth. Run from this folder after TSMini/prepare_rome.py:

    .venv/bin/python prepare_rome.py              # everything except cell embeddings
    .venv/bin/python prepare_rome.py --node2vec   # also train the cell embeddings (about 9 h on CPU)

With --dataset beijing it does the same for Beijing, after TSMini/prepare_beijing.py
(file names then start with beijing_20200).

Reads   ../TSMini/data/rome_20200.h5
        ../TSMini/data/rome_20200_traj_simi_dict_<measure>.pkl
Writes  data/rome_20200                           DataFrame with merc_seq, same row order
        data/rome_20200_traj_simi_dict_<measure>.pkl   copied unchanged
        data/rome_20200_newsimi_*.pkl             test sets for train.py
        data/rome_20200_cell100_*.pkl             cell space and embeddings (--node2vec)
"""
import argparse
import logging
import os
import shutil

import h5py
import pandas as pd

from config import Config
from utils.preprocessing_porto import generate_newsimi_test_dataset, init_cellspace
from utils.tool_funcs import lonlat2meters


def load_trajectories(h5_path):
    with h5py.File(h5_path, 'r') as f:
        n = int(f.attrs['n'])
        x_min, x_max, y_min, y_max = f.attrs['merc_range'][()].tolist()
        merc_seq = [f['/trajs_merc/{}'.format(i)][:].tolist() for i in range(n)]

    # Config's area is what the cell space is built from, and it must cover every point
    low = lonlat2meters(Config.min_lon, Config.min_lat)
    high = lonlat2meters(Config.max_lon, Config.max_lat)
    buffer = Config.cellspace_buffer
    assert low[0] - buffer <= x_min and x_max <= high[0] + buffer, 'x outside the area in config.py'
    assert low[1] - buffer <= y_min and y_max <= high[1] + buffer, 'y outside the area in config.py'

    too_long = sum(len(t) > Config.max_traj_len for t in merc_seq)
    assert too_long == 0, '{} trajectories exceed max_traj_len'.format(too_long)
    assert n >= int(10000 / 0.3) + 1, 'fine-tuning reads 10000 trajectories from int(0.7*n)'
    return pd.DataFrame({'trajlen': [len(t) for t in merc_seq], 'merc_seq': merc_seq})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='rome', choices=['rome', 'beijing'])
    ap.add_argument('--tsmini_data', default='../TSMini/data')
    ap.add_argument('--measures', nargs='+', default=['frechet'])
    ap.add_argument('--node2vec', action='store_true', help='train the cell embeddings (about 9 h on CPU)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    Config.update({'dataset': args.dataset})
    os.makedirs(os.path.dirname(Config.dataset_file), exist_ok=True)
    src = os.path.join(args.tsmini_data, Config.dataset_prefix)

    df = load_trajectories(src + '.h5')
    df.to_pickle(Config.dataset_file)
    logging.info('wrote %s (%d trajectories)', Config.dataset_file, len(df))

    for measure in args.measures:
        name = '_traj_simi_dict_{}.pkl'.format(measure)
        shutil.copyfile(src + name, Config.dataset_file + name)
        logging.info('copied %s', name)

    generate_newsimi_test_dataset()

    if args.node2vec:
        init_cellspace(df.merc_seq.tolist())
    logging.info('done')


if __name__ == '__main__':
    main()
