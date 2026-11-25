import os
import time
import logging
import pickle
import h5py
import math
import random
import os
import numpy as np
from torch.utils.data import Dataset
from config import Config


def _use_database():
    return (Config.dataset == 'beijing'
            or os.environ.get('TRAINING_DATA_SOURCE', '').lower() in ('db', 'database')
            or bool(os.environ.get('PGHOST')))


def _db_connection():
    import psycopg
    return psycopg.connect(
        host=os.environ.get('PGHOST', 'localhost'), port=os.environ.get('PGPORT', '5433'),
        user=os.environ.get('PGUSER', 'str'), password=os.environ.get('PGPASSWORD', 'str'),
        dbname=os.environ.get('PGDATABASE', 'trajectories'))


# Load traj dataset for trajsimi learning
def read_trajsimi_traj_dataset(file_path):
    logging.info('[trajsimi traj dataset] Start loading.')
    _time = time.time()

    if _use_database():
        prefix = os.path.basename(file_path)
        with _db_connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT n, merc_range FROM data.dataset WHERE name = %s', (prefix,))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f'Dataset {prefix!r} is not imported into data.dataset')
            n, merc_range = row
            merc_range = list(merc_range)
            offset_idx = int(n * 0.7)
            cur.execute("""SELECT idx, ST_X((dp).geom), ST_Y((dp).geom)
                           FROM (SELECT idx, ST_DumpPoints(geom) AS dp FROM data.trajectory
                                 WHERE dataset = %s AND idx >= %s AND idx < %s) p
                           ORDER BY idx, (dp).path[1]""", (prefix, offset_idx, offset_idx + 10000))
            grouped = {}
            for idx, x, y in cur:
                grouped.setdefault(idx, []).append([x, y])
        trajs = [grouped[i] for i in range(offset_idx, offset_idx + 10000)]
        if len(trajs) != 10000:
            raise RuntimeError(f'{prefix} has fewer than 10,000 trajectories in the TSMini training slice')
        return trajs[:7000], trajs[7000:8000], trajs[8000:], merc_range

    file_path += '.h5'
    datafile = h5py.File(file_path, 'r')
    merc_range = datafile.attrs['merc_range'][()].tolist()
    
    offset_idx = int(datafile.attrs['n'] * 0.7) # use eval dataset
    l = 10000
    trajs = []
    for i in range(l):
        idx = i + offset_idx
        trajs.append(datafile['/trajs_merc/%s' % idx][:].tolist())
    
    datafile.close()
    
    train_idx = (int(l*0), int(l*0.7))
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))
    trains = trajs[train_idx[0] : train_idx[1]]
    evals = trajs[eval_idx[0] : eval_idx[1]]
    tests = trajs[test_idx[0] : test_idx[1]]

    logging.info("[trajsimi traj dataset] Loaded. @={:.2f}. traj: #total={} (trains/evals/tests={}/{}/{})" \
                .format(time.time() - _time, l, len(trains), len(evals), len(tests)))
    return trains, evals, tests, merc_range


# Load simi dataset for trajsimi learning
def read_trajsimi_simi_dataset(file_path):
    logging.info('[trajsimi simi dataset] Start loading.')
    _time = time.time()
    if _use_database():
        base = os.path.basename(file_path)
        prefix, measure = base.split('_traj_simi_dict_', 1)
        measure = measure.removesuffix('.pkl')
        with _db_connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT max_distance FROM data.ground_truth_set WHERE dataset = %s AND measure = %s', (prefix, measure))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f'Ground truth {prefix}/{measure} is not imported into data.ground_truth_set')
            max_distance = row[0]
            parts = []
            for split, n_rows in (('train', 7000), ('eval', 1000), ('test', 2000)):
                cur.execute('SELECT distances FROM data.ground_truth WHERE dataset = %s AND measure = %s AND split = %s ORDER BY row', (prefix, measure, split))
                matrix = np.empty((n_rows, n_rows), dtype=np.float32)
                n_read = 0
                for row_idx, values in enumerate(cur):
                    if row_idx >= n_rows:
                        raise RuntimeError(f'Ground truth {prefix}/{measure}/{split} has more than {n_rows} rows')
                    matrix[row_idx] = values[0]
                    n_read += 1
                if n_read != n_rows:
                    raise RuntimeError(f'Ground truth {prefix}/{measure}/{split} has {n_read} rows; expected {n_rows}')
                parts.append(matrix)
        logging.info('[trajsimi simi dataset] Loaded from DB; rows=%s/%s/%s', *(len(p) for p in parts))
        return *parts, max_distance
    if not os.path.exists(file_path):
        logging.error('[trajsimi simi dataset does not exist')
        exit(200)

    with open(file_path, 'rb') as fh:
        trains_simi, evals_simi, tests_simi, max_distance = pickle.load(fh)
        logging.info("[trajsimi simi dataset] Loaded. @={:.2f}, trains/evals/tests={}/{}/{}" \
                .format(time.time() - _time, len(trains_simi), len(evals_simi), len(tests_simi)))
        return trains_simi, evals_simi, tests_simi, max_distance


class TrajSimiDatasetTraining(Dataset):
    def __init__(self, trains_traj, batchsize):
        self.trains_traj = trains_traj
        self.n = len(self.trains_traj)
        
        self.batchsize = batchsize
        self.niters = math.ceil( (self.n / batchsize)**2 )
    
    def __getitem__(self, index):
        sampled_idxs = random.sample(range(self.n), k = self.batchsize)
        trajs = [self.trains_traj[d_idx] for d_idx in sampled_idxs]

        return trajs, sampled_idxs
    
    def __len__(self):
        return self.niters

