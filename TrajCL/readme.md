# TrajCL: Contrastive Trajectory Similarity Learning with Dual-Feature Attention

------------------------------------------

Author's note in 2025:

If you're working on a graduation project or trying to compare a SOTA trajectory similarity method in a paper, I recommend checking out my latest work, [TSMini](https://github.com/changyanchuan/TSMini). It's not only methodologically simpler, but I also refactored a lot of the code in this repo, so it runs much faster with lower memory overall.

The experimental results are very strong — honestly better than many existing methods. With little luck, the paper just hasn’t been accepted yet. Thank you.

------------------------------------------


This is a pytorch implementation of the [TrajCL paper](https://arxiv.org/pdf/2210.05155.pdf):

```
@inproceedings{chang2023contrastive,
  title={Contrastive Trajectory Similarity Learning with Dual-Feature Attention},
  author={Chang, Yanchuan and Qi, Jianzhong and Liang, Yuxuan and Tanin, Egemen},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages={2933--2945},
  year={2023},
  organization={IEEE}
}
```


## Requirements
- Ubuntu 20.04 LTS with Python 3.7.7
- `pip install -r requirements.txt`
- Datasets can be downloaded from [here](https://drive.google.com/drive/folders/1wvFSdi4T1RvG1ww7TlobQJoTSBdJ7zWq?usp=sharing), and `tar -zxvf TrajCL_dataset.tar.gz -C ./data/`


## Quick Start
To train TrajCL and test it as a standalone trajectory measure (cf. Section V.B in paper):

```bash
python train.py --dataset porto
```

To fine-tune the pre-trained TrajCL to learn to approximate existing heuristic trajectory similarity measures (cf. Section V.F in paper):

(Prerequisites: a pre-trained TrajCL model, in other words, make sure you ran the last command once.)

```bash
python train_trajsimi.py --dataset porto --trajsimi_measure_fn_name hausdorff
```


## FAQ
#### Installation
It may occur failure while installing torch-geometric related packages, including torch-scatter, torch-sparse, torch-cluster and torch-spline-conv, when you use `pip install xxx` to install them directly. That is a commom issue for the older PyTorch versions. A solution can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Simply speaking, these package need to be installed from wheels. For example, `pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`.

#### Datasets
To use your own datasets, you may need to create your own pre-processing script like `./utils/preprocessing_porto.py`. Also, the MBR of the space is required to fill into `config.py`. (See `./utils/preprocessing_porto.py` and `config.py` for more details.)



## Contact
Email changyanchuan@gmail.com if you have any queries.

## Local changes in this project

Copied from upstream commit `fbc98f5`. This setup targets WSL on CPU, with Python 3.11:

```bash
uv venv --python 3.11 --prompt TrajCL .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
.venv/bin/python smoke_test.py    # synthetic data, checks that everything runs
```

### Training on Rome

Uses TSMini's trajectories, splits and Frechet ground truth, so run `TSMini/prepare_rome.py` first.
Times are estimates for an 8-core laptop CPU. Each step needs the one before it.

```bash
.venv/bin/python prepare_rome.py             # dataset files, under a minute
.venv/bin/python prepare_rome.py --node2vec  # cell embeddings, about 9 h
.venv/bin/python train.py --dataset rome     # pre-training, about 1.4 h
.venv/bin/python train_trajsimi.py --dataset rome --trajsimi_measure_fn_name frechet  # fine-tuning, up to 35 h
```

The `--trajsimi_measure_fn_name frechet` flag is required, as the default in `config.py` is EDwP.

### Training on Beijing

Same steps with `beijing` instead of `rome`. The trajectories come from the shared aSTEP database
(see "Beijing data for TSMini and TrajCL" in the root README), so build TSMini's Beijing files first.
The area is about twice the size of Rome's, so expect node2vec to take about twice as long.

```bash
.venv/bin/python prepare_beijing.py
.venv/bin/python prepare_beijing.py --node2vec
.venv/bin/python train.py --dataset beijing
.venv/bin/python train_trajsimi.py --dataset beijing --trajsimi_measure_fn_name frechet
```

### Saving results so nothing is trained twice

After each long step, save its output in the project database (schema `trajcl`, from the repository root).
That stores the node2vec cell embeddings, all `.pt` models, the logs, and (after `embed.py`) an embedding per trajectory:

```bash
.venv/bin/python embed.py --dataset beijing --trajsimi_measure_fn_name frechet   # after fine-tuning, a few minutes
docker compose run --rm --build trajcl-import --prefix beijing_20200             # from the repository root
```

To continue on a machine without the files (e.g. skip node2vec or pre-training), write them back to disk:

```bash
docker compose run --rm --build trajcl-import --prefix beijing_20200 --restore
```

Similar trajectories by embedding: `SELECT * FROM trajcl.similar('beijing_20200', 5);`

### Differences from upstream

Changes that can affect results:

- `model/dual_attention.py`: `repeat` is now `repeat_interleave`. Upstream gave attention heads the spatial masks of other trajectories in the batch, so embeddings depended on batching.
- `model/node2vec_.py`, `utils/preprocessing_porto.py`: `init_cellspace` can start walks only from cells that trajectories visit, which Rome uses. Walks can still enter unvisited cells (about 74% of walk steps on Rome), so those cells get some training too.
- `utils/data_loader.py`: pre-training uses at most the first 70% of trajectories, so small datasets don't train on the eval and test slices. Porto is unaffected.
- `requirements.txt`: torch 2.1.1 and current libraries replace Python 3.7 and torch 1.8.1.

Compatibility and speed, with no intended effect on results:

- `model/moco.py`: removed a hard-coded `.cuda()`.
- `utils/tool_funcs.py`: NVIDIA monitoring is optional.
- `utils/data_loader.py`: `pickle` replaces `pickle5`.
- `model/node2vec_.py`: saved cell embeddings are detached from autograd.
- `model/trajcl.py`: removed `set_detect_anomaly`, which made batches about 8 times slower. `test()` sets database sizes as fractions of the database.
- `utils/preprocessing_porto.py`: `traj_dist` is optional, `iteritems` is now `items`, and a `num_cores` crash is fixed.
- `config.py`: added the `rome` dataset.
- Added `prepare_rome.py` and `smoke_test.py`.
