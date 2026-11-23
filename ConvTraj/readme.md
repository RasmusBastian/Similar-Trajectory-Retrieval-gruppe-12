# Revisiting CNNs for Trajectory Similarity Learning (VLDB 2025)

The code officially implements the ConvTraj algorithm for trajectory distance computation approximation.

## datasets
The overall dataset and groundtruth can be found https://huggingface.co/datasets/Airchang/SDD

The Geolife dataset and groundtruth can be found https://drive.google.com/file/d/1mVQgXcXoCuICZFF5N3ySaz0fwGclQO-P/view?usp=drive_link.

The Porto dataset and groundtruth can be found https://drive.google.com/file/d/1s_XGg0hq-eOqGgohHfpEqOW65_W2uuS7/view?usp=sharing (Due to disk space limitations, only the DTW distance is included in this link. If you want to get other distance groundtruth, please contact me.)

The TrajCl-Porto dataset can be found https://drive.google.com/file/d/1SZJEBF3PUi8EAlx20l_0-jYTR0ksfJjw/view?usp=sharing

## Contributors

- Zhihao Chang: changzhihao@zju.edu.cn



## Reference
If you want to use this code, please cite:
```
@misc{chang2024revisiting,
      title={Revisiting CNNs for Trajectory Similarity Learning}, 
      author={Zhihao Chang and Linzhu Yu and Huan Li and Sai Wu and Gang Chen and Dongxiang Zhang},
      year={2024},
      eprint={2405.19761},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Local changes in this project

Copied from upstream commit `dd4f1f1` (May 2025). Upstream has no license file and no requirements file.
Setup (Python 3.10–3.12; Python 3.13+ has no torch 2.5 wheels):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python smoke_test.py    # synthetic data, checks that everything runs (~30 s)
```

### Training on Rome or Beijing

Uses TSMini's trajectories, splits and Frechet ground truth, so `../TSMini/data/<prefix>.h5` and
`<prefix>_traj_simi_dict_frechet.pkl` must exist (run `TSMini/prepare_rome.py`, or unpack the team's shared zip).

```bash
.venv/bin/python train_pipeline.py --dataset beijing            # prepare -> train (200 epochs) -> embed
.venv/bin/python train_pipeline.py --dataset beijing --from train --epochs 50
```

Or step by step:

```bash
.venv/bin/python prepare.py --dataset beijing      # data/beijing_20200/, under a minute
.venv/bin/python train.py --train_flag beijing_20200_frechet --dist_type frechet --epoch_num 200 --test_epoch 200 \
    --root_read_path data/beijing_20200 --root_write_path exp/beijing_20200
.venv/bin/python embed.py --dataset beijing        # exp/embeddings_beijing_20200.npz + HR@5/HR@20/R5@20
```

How the data maps onto ConvTraj (details in `prepare.py`):

- train = TSMini's 7000 train trajectories, query = TSMini's 2000 test trajectories, base = all others.
- Coordinates go from TSMini's Web-Mercator metres back to lon/lat, because ConvTraj's grid is in degrees (0.002° cells).
- Frechet distances are divided by `max_distance`, like TSMini does.
- `embed.py` evaluates on the 2000 test trajectories with the same `hitting_ratio` as TSMini and TrajCL,
  so the numbers are directly comparable. Embeddings are compared with L2 (TrajCL uses L1).
- `exp/embeddings_<prefix>.npz` has row i = h5 trajectory i (= `tsmini.trajectory.idx`), same format as TrajCL's.

### Differences from upstream

- `train.py`: falls back to Apple MPS or CPU instead of exiting without CUDA.
- `tools/function.py`: `set_dataset` reads split sizes from `meta.json`; `initialize_model` uses ConvTraj for any
  dataset folder (upstream only matched its own folder names).
- `tools/pre_rep.py`: `image_encode` returns one float32 array instead of nested Python lists (same values,
  far less memory; Beijing's grid is ~175 x 140 cells per trajectory).
- `mynn/Traj_KNN.py`: checkpoints load with `map_location`; the query x base test matrix is optional (it is never
  used in training); `extract_feature_from_path` no longer crashes on fewer than 10,240 trajectories.
- `config/new_config.py`: creates nested output folders.
- Added `prepare.py`, `embed.py`, `train_pipeline.py`, `smoke_test.py`, `requirements.txt`.

### Possible future change: pick the best epoch on the eval split

Not done yet (noted 23/11-2026). Upstream ConvTraj trains a fixed number of epochs and keeps the model from the
last epoch (`mynn/Traj_KNN.py`, saved every `--test_epoch`). TSMini and TrajCL instead evaluate on the 1000-trajectory
eval split after every epoch, keep the epoch with the best HR (`*_best.pt`) and stop early when it stops improving
(`task/trajsimi.py` in both, as in their upstream code). So ConvTraj is currently at a small disadvantage in the comparison.

To make it equal: add TSMini's eval split to `prepare.py` (with its ground truth from the pkl), evaluate HR@5 with
`embed.hitting_ratio` after each epoch in `NeuTrajTrainer.train`, and save only when it improves. Trade-off: the
comparison becomes fairer, but ConvTraj is then no longer run exactly as its authors did, so list it under
"Differences from upstream" if it is added.

Beijing results without this change (200 epochs, last epoch): HR@5 0.747, HR@20 0.796, R5@20 0.992.
