# Training TSMini on Beijing (for Jesper)

**Goal:** a fair TSMini vs TrajCL comparison. TrajCL is trained on Beijing (your run from 14 Nov), but TSMini is
only trained on Rome, and for just 1 epoch, so the two can't be compared yet. This run trains TSMini on **the same
Beijing data** (same 49,934 trajectories, same Fréchet ground truth, same 7000/1000/2000 split) with **the same
budget as TrajCL's fine-tuning: up to 30 epochs**. It stops early if the eval score hasn't improved for 5 epochs.

Only train **TSMini on Beijing**. File names are fixed, so if anyone else trains the same model, the checkpoints overwrite each other.

## 1. Get the code and data

```bash
git fetch && git switch tsmini-beijing          # this branch: adds --trajsimi_epoch to TSMini
python scripts/share_data.py unpack <path>/beijing_20200_20261114-1424_jesper.zip   # if your copy is gone
```

You need `TSMini/data/beijing_20200.h5` and `TSMini/data/beijing_20200_traj_simi_dict_frechet.pkl` (both in that zip).

## 2. A GPU environment for TSMini

Your TrajCL `.venv-gpu` won't do, because TSMini's packages differ. From `TSMini/`:

```bash
uv venv .venv-gpu --python 3.11
uv pip install --python .venv-gpu/bin/python -r requirements.txt   # torch==2.1.1 from PyPI is the CUDA build
.venv-gpu/bin/python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## 3. Train, embed, pack

From `TSMini/`:

```bash
.venv-gpu/bin/python train_trajsimi.py --dataset beijing --trajsimi_measure_fn_name frechet --trajsimi_epoch 30
.venv-gpu/bin/python search_rome.py embed --dataset beijing
cd .. && python scripts/share_data.py pack --prefix beijing_20200
```

| Step | Writes |
|---|---|
| train | `TSMini/exp/snapshots/beijing_20200_trajsimi_TSMini_frechet_best.pt`, `TSMini/exp/log/<timestamp>.log` |
| embed | `TSMini/exp/embeddings_beijing_20200.npz` (49,934 × 128) |
| pack | `shared/beijing_20200_<date>_<name>.zip`, which also contains your TrajCL models and everything else for Beijing |

**How long it takes:** on CPU, one Rome epoch took about 22 minutes. On your GPU, expect well under that per epoch, so the
whole run is probably 1–3 hours. It ends early if 5 epochs in a row bring no improvement.

## 4. Check that the run went well

At the end of the log (`TSMini/exp/log/<timestamp>.log`):

- `training end. ... best_epoch=<n>, best_hr_eval=<x>`. If `best_epoch` is 29 (the last one), the model was still
  improving when it stopped. Say so when you share it.
- `test. ... hr=...` and a final `[EXPFlag]model=TSMini,dataset=beijing_20200,...,hr5=...,hr20=...,hr20in5=...`.
  These are the test-set numbers the comparison uses.

For reference, TrajCL on Beijing scored hr5 = 0.534, hr20 = 0.652 and hr20in5 = 0.889.

## 5. Hand it over

Upload the zip to the team Drive and tell Rasmus. He'll then run:

```bash
python scripts/share_data.py unpack shared/<your zip>
docker compose run --rm --build tsmini-import --prefix beijing_20200      # no --replace: it adds the embeddings
TrajCL/.venv/bin/python scripts/compare_models.py --prefix beijing_20200 --nq 100 --query 5
```

`compare_models.py` runs both models on the same random queries, scores them against exact Fréchet search and
writes `results/beijing_20200/comparison.json`, `comparison.png` and `query_5.png`.
