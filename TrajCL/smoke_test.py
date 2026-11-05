#!/usr/bin/env python3
"""
Checks that TrajCL runs on the installed dependencies, using synthetic data in a
temp folder. It says nothing about model quality. Run from this folder:

    .venv/bin/python smoke_test.py

Covers Node2Vec cell embeddings, pre-training and its test, checkpoint reload,
padding handling, EDwP ground truth, and fine-tuning. Only the file readers in
TrajSimi.load_trajsimi_dataset are bypassed, as they require 10,000 trajectories.
"""
import logging
import os
import pickle
import random
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from config import Config
from model.trajcl import TrajCL, TrajCLTrainer, collate_for_test
from task.trajsimi import TrajSimi
from utils.edwp import edwp
from utils.preprocessing_porto import _get_simi_fn, _simi_matrix, init_cellspace
from utils.tool_funcs import lonlat2meters


def synthetic_traj(rng, x_min, y_min, x_max, y_max):
    """A walk with momentum that bounces off the area edge and spans several cells."""
    x, y = rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)
    heading = rng.uniform(0, 2 * np.pi)
    traj = []
    for _ in range(rng.randint(25, 40)):
        heading += rng.gauss(0, 0.3)
        x += 150 * np.cos(heading)
        y += 150 * np.sin(heading)
        if not x_min <= x <= x_max or not y_min <= y <= y_max:
            heading += np.pi
            x, y = min(max(x, x_min), x_max), min(max(y, y_min), y_max)
        traj.append([float(x), float(y)]) # plain floats, as in the real datasets
    return traj


def check_finite(model, label):
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            assert torch.isfinite(value).all(), '{}: {} has non-finite values'.format(label, name)


def main(tmp):
    os.makedirs(tmp + '/data')
    Config.root_dir = tmp
    Config.checkpoint_dir = tmp
    # Node2Vec's 20 epochs are hard-coded, so large cells keep the graph small and the test short
    Config.update({'dataset': 'porto', 'seed': 2000, 'cell_size': 400.0})
    # a 2 km area, which replaces the full Porto area set by Config.update
    Config.min_lon, Config.max_lon = -8.62, -8.60
    Config.min_lat, Config.max_lat = 41.14, 41.16
    Config.trajcl_batch_size = 16
    Config.trajcl_training_epochs = 2
    Config.trajsimi_batch_size = 16
    Config.trajsimi_epoch = 2
    Config.trajsimi_measure_fn_name = 'edwp'

    # inside the cell space, which adds a buffer around this area
    box = (*lonlat2meters(Config.min_lon, Config.min_lat), *lonlat2meters(Config.max_lon, Config.max_lat))
    rng = random.Random(2000)
    trajs = [synthetic_traj(rng, *box) for _ in range(250)]

    init_cellspace(trajs)
    with open(Config.dataset_cell_file, 'rb') as fh:
        cs = pickle.load(fh)
    with open(Config.dataset_embs_file, 'rb') as fh:
        embs = pickle.load(fh)
    assert embs.shape == (cs.x_size * cs.y_size, Config.cell_embedding_dim), embs.shape
    assert not embs.requires_grad and torch.isfinite(embs).all()
    logging.info('OK node2vec: %s cell embeddings', tuple(embs.shape))

    pd.DataFrame({'merc_seq': trajs}).to_pickle(Config.dataset_file)

    # the file names and layout TrajCLTrainer.test() reads
    queries, database = trajs[:8], trajs[8:40]
    names = ['raw'] + [vt + '_' + str(r) for vt in ('downsampling', 'distort') for r in (0.1, 0.2, 0.3, 0.4, 0.5)]
    for name in names:
        with open('{}_newsimi_{}.pkl'.format(Config.dataset_file, name), 'wb') as fh:
            pickle.dump((queries, database), fh)

    trainer = TrajCLTrainer(Config.trajcl_aug1, Config.trajcl_aug2)
    before = {k: v.clone() for k, v in trainer.model.clmodel.encoder_q.state_dict().items()}
    trainer.train()
    after = trainer.model.clmodel.encoder_q.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), 'training changed no weights'
    check_finite(trainer.model, 'pre-training')
    trainer.test()
    logging.info('OK pre-training and test')

    trainer.model.eval()
    with torch.no_grad():
        batch = collate_for_test(trajs[:16], cs, trainer.embs)
        emb = trainer.model.interpret(*batch)
        assert emb.shape == (16, Config.seq_embedding_dim), emb.shape
        assert torch.isfinite(emb).all()

        reloaded = TrajCL().load_checkpoint().to(Config.device).eval()
        assert torch.allclose(reloaded.interpret(*batch), emb, atol=1e-5), 'reload changed outputs'
        logging.info('OK checkpoint reload')

        # A trajectory's embedding must not depend on how much padding its batch needs.
        shortest = min(range(16), key=lambda i: batch[2][i].item())
        assert batch[2][shortest] < batch[2].max(), 'test needs trajectories of unequal length'
        alone = trainer.model.interpret(*collate_for_test([trajs[shortest]], cs, trainer.embs))
        diff = (alone[0] - emb[shortest]).abs().max().item()
        assert diff < 1e-4, 'padding changes the embedding, max diff {}'.format(diff)
        logging.info('OK padding invariance: max diff %.2e', diff)

    assert edwp(np.array(trajs[0]), np.array(trajs[0])) == 0
    assert edwp(np.array(trajs[0]), np.array(trajs[1])) > 0
    frames = [pd.DataFrame({'merc_seq': trajs[i:i + 50]}) for i in (0, 50, 100)]
    simi = [_simi_matrix(_get_simi_fn('edwp'), df) for df in frames]
    max_distance = max(max(map(max, m)) for m in simi)
    assert max_distance > 0
    logging.info('OK EDwP ground truth')

    values = [df.merc_seq.values for df in frames]
    dataset = {'trains_traj': values[0], 'evals_traj': values[1], 'tests_traj': values[2],
               'trains_simi': simi[0], 'evals_simi': simi[1], 'tests_simi': simi[2],
               'max_distance': max_distance}
    TrajSimi.load_trajsimi_dataset = staticmethod(lambda: dataset)

    encoder = TrajCL().load_checkpoint().to(Config.device)
    metrics = TrajSimi(encoder).train()
    for key in ('hr5', 'hr20', 'hr20in5'):
        assert 0 <= metrics[key] <= 1, (key, metrics[key])
    check_finite(encoder, 'fine-tuning')
    logging.info('OK fine-tuning: %s', {k: round(metrics[k], 3) for k in ('hr5', 'hr20', 'hr20in5')})

    logging.info('SMOKE TEST PASSED')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    tmp = tempfile.mkdtemp(prefix='trajcl_smoke_')
    try:
        main(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
