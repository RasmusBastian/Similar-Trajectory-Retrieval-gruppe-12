import os
import re
import time
import logging
import pickle
import numpy as np
import pandas as pd
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


def _database_trajectories(prefix, start=0, stop=None):
    import pandas as pd
    with _db_connection() as conn, conn.cursor() as cur:
        cur.execute('SELECT n FROM tsmini.dataset WHERE name = %s', (prefix,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f'Dataset {prefix!r} is not imported into tsmini.dataset')
        end = min(row[0], stop if stop is not None else row[0])
        cur.execute("""SELECT idx, (dp).path[1] AS point_no, ST_X((dp).geom), ST_Y((dp).geom)
                       FROM (SELECT idx, ST_DumpPoints(geom) AS dp FROM tsmini.trajectory
                             WHERE dataset = %s AND idx >= %s AND idx < %s) q
                       ORDER BY idx, point_no""", (prefix, start, end))
        grouped = {}
        for idx, _, x, y in cur:
            grouped.setdefault(idx, []).append([x, y])
    return pd.DataFrame({'merc_seq': [grouped[i] for i in sorted(grouped)]}, index=sorted(grouped))


def load_pickle(path):
    """Load a pickle input from trajcl.file when database mode is enabled."""
    if _use_database():
        import pickle
        name = os.path.basename(path)
        test_match = re.fullmatch(r'(.+)_newsimi_(raw|downsampling_[0-9.]+|distort_[0-9.]+)\.pkl', name)
        if test_match:
            prefix, variant_name = test_match.groups()
            if variant_name == 'raw':
                variant, rate = 'raw', 0.0
            else:
                variant, rate_text = variant_name.rsplit('_', 1)
                rate = float(rate_text)
            with _db_connection() as conn, conn.cursor() as cur:
                cur.execute('SELECT role, idx, points FROM trajcl.test_trajectory WHERE dataset = %s AND variant = %s AND rate = %s ORDER BY role, idx',
                            (prefix, variant, rate))
                grouped = {'query': [], 'database': []}
                for role, idx, points in cur.fetchall():
                    grouped[role].append((idx, points))
            if not grouped['query'] and not grouped['database']:
                raise RuntimeError(f'Test inputs {prefix}/{variant}/{rate:g} are not imported into trajcl.test_trajectory')
            return [[points for _, points in grouped[role]] for role in ('query', 'database')]
        rel = 'data/' + name
        with _db_connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT content FROM trajcl.file WHERE path = %s', (rel,))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(f'{rel} is not imported into trajcl.file')
        return pickle.loads(row[0])
    import pickle
    with open(path, 'rb') as fh:
        return pickle.load(fh)

# 1) read raw pd, 2) split into 3 partitions
def read_traj_dataset(file_path):
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    if _use_database():
        prefix = os.path.basename(file_path)
        with _db_connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT n FROM tsmini.dataset WHERE name = %s', (prefix,))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f'Dataset {prefix!r} is not imported into tsmini.dataset')
            l = row[0]
        train_count = min(200000, int(l * 0.7))
        train_data = _database_trajectories(prefix, 0, train_count)
        empty = pd.DataFrame({'merc_seq': []})
        logging.info('[Load traj dataset] END. @=%.0f, #=%d (train=%d, DB source)', time.time() - _time, l, len(train_data))
        return TrajDataset(train_data), TrajDataset(empty), TrajDataset(empty)
    else:
        trajs = pd.read_pickle(file_path)

    l = trajs.shape[0]
    train_idx = (int(l*0), min(200000, int(l*0.7))) # never reach the eval/test slices on small datasets
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))

    _train = TrajDataset(trajs[train_idx[0]: train_idx[1]])
    _eval = TrajDataset(trajs[eval_idx[0]: eval_idx[1]])
    _test = TrajDataset(trajs[test_idx[0]: test_idx[1]])

    logging.info('[Load traj dataset] END. @={:.0f}, #={}({}/{}/{})' \
                .format(time.time() - _time, l, len(_train), len(_eval), len(_test)))
    return _train, _eval, _test


class TrajDataset(Dataset):
    def __init__(self, data):
        # data: DataFrame
        self.data = data

    def __getitem__(self, index):
        return self.data.loc[index].merc_seq

    def __len__(self):
        return self.data.shape[0]


# Load traj dataset for trajsimi learning
def read_trajsimi_traj_dataset(file_path):
    logging.info('[Load trajsimi traj dataset] START.')
    _time = time.time()

    db_mode = _use_database()
    if db_mode:
        prefix = os.path.basename(file_path)
        with _db_connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT n FROM tsmini.dataset WHERE name = %s', (prefix,))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f'Dataset {prefix!r} is not imported into tsmini.dataset')
            offset_idx = int(row[0] * 0.7)
        df_trajs = _database_trajectories(prefix, offset_idx, offset_idx + 10000)
    else:
        df_trajs = pd.read_pickle(file_path)
    offset_idx = 0 if db_mode else int(df_trajs.shape[0] * 0.7) # database query already selected the eval slice
    df_trajs = df_trajs.iloc[offset_idx : offset_idx + 10000]
    assert df_trajs.shape[0] == 10000
    l = 10000

    train_idx = (int(l*0), int(l*0.7))
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))
    trains = df_trajs.iloc[train_idx[0] : train_idx[1]]
    evals = df_trajs.iloc[eval_idx[0] : eval_idx[1]]
    tests = df_trajs.iloc[test_idx[0] : test_idx[1]]

    logging.info("trajsimi traj dataset sizes. traj: #total={} (trains/evals/tests={}/{}/{})" \
                .format(l, trains.shape[0], evals.shape[0], tests.shape[0]))
    return trains, evals, tests


# Load simi dataset for trajsimi learning
def read_trajsimi_simi_dataset(file_path):
    logging.info('[Load trajsimi simi dataset] START.')
    _time = time.time()
    if _use_database():
        base = os.path.basename(file_path)
        prefix, measure = base.split('_traj_simi_dict_', 1)
        measure = measure.removesuffix('.pkl')
        with _db_connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT max_distance FROM tsmini.ground_truth_set WHERE dataset = %s AND measure = %s', (prefix, measure))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f'Ground truth {prefix}/{measure} is not imported into tsmini.ground_truth_set')
            max_distance = row[0]
            parts = []
            for split, n_rows in (('train', 7000), ('eval', 1000), ('test', 2000)):
                cur.execute('SELECT distances FROM tsmini.ground_truth WHERE dataset = %s AND measure = %s AND split = %s ORDER BY row', (prefix, measure, split))
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
        return *parts, max_distance
    if not os.path.exists(file_path):
        logging.error('trajsimi simi dataset does not exist')
        exit(200)

    with open(file_path, 'rb') as fh:
        trains_simi, evals_simi, tests_simi, max_distance = pickle.load(fh)
        logging.info("[trajsimi simi dataset loaded] @={}, trains/evals/tests={}/{}/{}" \
                .format(time.time() - _time, len(trains_simi), len(evals_simi), len(tests_simi)))
        return trains_simi, evals_simi, tests_simi, max_distance
