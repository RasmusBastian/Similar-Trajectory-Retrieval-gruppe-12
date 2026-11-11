#!/usr/bin/env python3
"""
Build TrajCL's Beijing dataset from TSMini's, exactly like prepare_rome.py does for Rome.
Run from this folder after TSMini has its Beijing files (see ../TSMini/export_beijing.py):

    .venv/bin/python prepare_beijing.py              # everything except cell embeddings
    .venv/bin/python prepare_beijing.py --node2vec   # also train the cell embeddings
                                                     # (area is about 2x Rome's, so expect ~2x the time)

Reads   ../TSMini/data/beijing_20200.h5
        ../TSMini/data/beijing_20200_traj_simi_dict_<measure>.pkl
Writes  data/beijing_20200*                      same files as prepare_rome.py writes for Rome
"""
from prepare_rome import main

if __name__ == '__main__':
    main('beijing')
