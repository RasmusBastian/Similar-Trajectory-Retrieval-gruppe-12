#!/usr/bin/env python3
"""
Embed every trajectory of a dataset with a fine-tuned TrajCL model (after train_trajsimi.py),
so similarity search does not need the model any more: the L1 distance between two embeddings
approximates the measure the model was fine-tuned on. import_to_db.py stores the result in trajcl.embedding.

    .venv/bin/python embed.py --dataset beijing --trajsimi_measure_fn_name frechet

Reads   exp/snapshots/<prefix>_trajsimi_TrajCL_<measure>_best.pt   fine-tuned model
        data/<prefix>, data/<prefix>_cell100_cellspace.pkl, data/<prefix>_cell100_embdim*_embs.pkl
Writes  exp/embeddings_<prefix>.npz   emb: [n, 256] float32, row i = trajectory i (= tsmini.trajectory.idx),
                                      model: checkpoint file name
"""
import argparse
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from config import Config
from model.trajcl import TrajCL
from task.trajsimi import TrajSimi, TrajSimiRegression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--trajsimi_measure_fn_name', default='frechet')
    ap.add_argument('--dumpfile_uniqueid', default='')
    args = ap.parse_args()
    Config.update(vars(args))

    ckpt = '{}/{}_trajsimi_{}_{}_best{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix,
                                                   Config.trajsimi_encoder_name, Config.trajsimi_measure_fn_name,
                                                   Config.dumpfile_uniqueid)
    if not os.path.isfile(ckpt):
        sys.exit('fine-tuned model not found: {} (run train_trajsimi.py, or import_to_db.py --restore)'.format(ckpt))

    state = torch.load(ckpt, map_location=Config.device)
    encoder = TrajCL()
    encoder.clmodel.encoder_q.load_state_dict(state['encoder_q'])
    encoder.to(Config.device).eval()
    head = TrajSimiRegression(Config.seq_embedding_dim)
    head.load_state_dict(state['trajsimi'])
    head.to(Config.device).eval()

    # the batch generator only needs the cell space and cell embeddings from its TrajSimi instance
    ctx = SimpleNamespace(cellspace=pickle.load(open(Config.dataset_cell_file, 'rb')),
                          cellembs=pickle.load(open(Config.dataset_embs_file, 'rb')).to(Config.device))
    trajs = pd.read_pickle(Config.dataset_file).merc_seq.values

    t0 = time.time()
    outs = []
    with torch.no_grad():
        for i, (cells, spatial, lens) in enumerate(TrajSimi.trajsimi_dataset_generator_single_batchi(ctx, trajs)):
            outs.append(head(encoder.interpret(cells, spatial, lens)).cpu())
            if i % 20 == 0:
                print('  embedded {}/{}'.format(min((i + 1) * Config.trajsimi_batch_size, len(trajs)), len(trajs)),
                      flush=True)
    emb = torch.cat(outs).numpy().astype(np.float32)

    out = '{}/exp/embeddings_{}.npz'.format(Config.root_dir, Config.dataset_prefix)
    np.savez(out, emb=emb, model=os.path.basename(ckpt))
    print('wrote {} ({} x {}) in {:.0f}s'.format(out, *emb.shape, time.time() - t0))


if __name__ == '__main__':
    main()
