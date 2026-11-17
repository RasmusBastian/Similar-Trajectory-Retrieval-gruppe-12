import os
import time
import math
import random
import numpy as np
from itertools import tee
from pynvml import *
import psutil
from sklearn.preprocessing import label_binarize
from datetime import datetime, timezone, timedelta


def _database_mode():
    from config import Config
    return (Config.dataset == 'beijing'
            or os.environ.get('TRAINING_DATA_SOURCE', '').lower() in ('db', 'database')
            or bool(os.environ.get('PGHOST')))

try:
    nvmlInit() # need initializztion here
    _NVML_OK = True
except Exception: # no NVIDIA driver / GPU
    _NVML_OK = False


def load_cell_embs(path):
    """Read node2vec cell embeddings (a pickled tensor) onto the CPU, also when they were saved on a GPU.
    Plain pickle.load fails without CUDA for a tensor that was on a GPU when it was pickled."""
    import io
    import pickle
    import torch

    class _CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    if _database_mode():
        import psycopg
        with psycopg.connect(
            host=os.environ.get('PGHOST', 'localhost'), port=os.environ.get('PGPORT', '5433'),
            user=os.environ.get('PGUSER', 'str'), password=os.environ.get('PGPASSWORD', 'str'),
            dbname=os.environ.get('PGDATABASE', 'trajectories')) as conn, conn.cursor() as cur:
            cur.execute('SELECT content FROM trajcl.file WHERE path = %s', ('data/' + os.path.basename(path),))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"data/{os.path.basename(path)} is not imported into trajcl.file")
        return _CPUUnpickler(io.BytesIO(row[0])).load()
    with open(path, 'rb') as fh:
        return _CPUUnpickler(fh).load()


def load_checkpoint(path, map_location):
    """Load a trained TrajCL checkpoint from PostgreSQL in database mode."""
    import io
    import torch
    if _database_mode():
        import psycopg
        rel = 'exp/snapshots/' + os.path.basename(path)
        with psycopg.connect(
            host=os.environ.get('PGHOST', 'localhost'), port=os.environ.get('PGPORT', '5433'),
            user=os.environ.get('PGUSER', 'str'), password=os.environ.get('PGPASSWORD', 'str'),
            dbname=os.environ.get('PGDATABASE', 'trajectories')) as conn, conn.cursor() as cur:
            cur.execute('SELECT content FROM trajcl.file WHERE path = %s', (rel,))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(f'{rel} is not imported into trajcl.file')
        return torch.load(io.BytesIO(row[0]), map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location, weights_only=False)


def save_embeddings(dataset, model, embeddings):
    """Persist freshly computed embeddings directly to PostgreSQL in database mode."""
    import psycopg
    with psycopg.connect(
        host=os.environ.get('PGHOST', 'localhost'), port=os.environ.get('PGPORT', '5433'),
        user=os.environ.get('PGUSER', 'str'), password=os.environ.get('PGPASSWORD', 'str'),
        dbname=os.environ.get('PGDATABASE', 'trajectories')) as conn, conn.cursor() as cur:
        cur.execute('DELETE FROM trajcl.embedding WHERE dataset = %s AND model = %s', (dataset, model))
        with cur.copy('COPY trajcl.embedding (dataset, model, idx, embedding) FROM STDIN WITH (FORMAT binary)') as copy:
            copy.set_types(['text', 'text', 'int4', 'float4[]'])
            for idx, vector in enumerate(embeddings):
                copy.write_row((dataset, model, idx, vector.tolist()))


def mean(x):
    if x == []:
        return 0.0
    return sum(x) / len(x)


def std(x):
    return np.std(x)


def minmax_norm(v, minv, maxv):
    return (v-minv) / (maxv-minv) + 1


def truncated_rand(mu = 0, sigma = 0.5, factor = 100, bound_lo = -100, bound_hi = 100):
    # using the defaults parameters, the success rate of one-pass random number generation is ~96%
    # gauss visualization: https://www.desmos.com/calculator/jxzs8fz9qr?lang=zh-CN
    while True:
        n = random.gauss(mu, sigma) * factor
        if bound_lo <= n <= bound_hi:
            break
    return n


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def l2_distance(lon1, lat1, lon2, lat2):
    return math.sqrt( (lon2 - lon1) ** 2 + (lat2 - lat1) ** 2 )


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6367 * 1000


# distance between two points on Earth using their latitude and longitude
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367 * 1000


# radian of the segment
def radian(lon1, lat1, lon2, lat2):
    dy = lat2 - lat1
    dx = lon2 - lon1
    r = 0.0

    if dx == 0:
        if dy >= 0:
            r = 1.5707963267948966 # math.pi / 2
        else: 
            r = 4.71238898038469 # math.pi * 1.5
        return round(r, 3)
    
    r = math.atan(dy / dx)
    # angle_in_degrees = math.degrees(angle_in_radians)
    if dx < 0:
        r = r + 3.141592653589793
    else:
        if dy < 0:
            r = r + 6.283185307179586
        else:
            pass
    return round(r, 3)


# ref: TrjSR
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))


# code ref: TrjSR
def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat


def dump_config_to_strs(file_path):
    # return list
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('from') or line.startswith('import') or line.startswith('#') \
                    or line.strip() == '':
                continue
            lines.append(line.strip())
    return lines


def log_file_name():
    dt = datetime.now(timezone(timedelta(hours=8)))
    return dt.strftime("%Y%m%d_%H%M%S") + '.log'


# torch: number of parameters 
def num_of_model_params(models):
    n = 0
    if type(models) == list:
        for model in models:
            n += sum(p.numel() for p in model.parameters() if p.requires_grad) 
    else:
        n = sum(p.numel() for p in models.parameters() if p.requires_grad)
    return n


class Metrics:
    def __init__(self):
        self.dic = {} # {a:[], b:[], ...}
   
    def __str__(self) -> str:
        s = ''
        kvs = list(self.mean().items()) # + list(self.std().items())
        for i, (k, v) in enumerate(kvs):
            if i != 0:
                s += ','
            s = s + '{}={:.6f}'.format(k, v)
        return s

    def add(self, d_: dict):
        for k, v in d_.items():
            self.dic[k] = self.dic.get(k, []) + [v]
    
    def get(self, k):
        return self.dic.get(k, [])
    
    def mean(self, k = None):
        if type(k) == str:
            return mean(self.get(k)) # division by zero
        dic_mean = {}
        for k, v in self.dic.items():
            dic_mean[k] = mean(v)
        return dic_mean

    def std(self, k = None):
        if type(k) == str:
            return std(self.get(k)) # division by zero
        dic_mean = {}
        for k, v in self.dic.items():
            dic_mean[k] = std(v)
        return dic_mean
    

class Timer(object):
    _instance = None
    _last_t = 0.0

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):
        self._last_t = time.time()

    def tick(self):
        _t = time.time()
        _delta = _t - self._last_t
        self._last_t = _t
        return _delta

timer = Timer()


class GPUInfo:

    _h = nvmlDeviceGetHandleByIndex(0) if _NVML_OK else None

    @classmethod
    def mem(cls):
        if cls._h is None:
            return 0, 0
        info = nvmlDeviceGetMemoryInfo(cls._h)
        return info.used // 1048576, info.total // 1048576 # in MB

class RAMInfo:
    @classmethod
    def mem(cls):
        return int(psutil.Process(os.getpid()).memory_info().rss / 1048576) # in MB
