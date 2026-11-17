# Similar Trajectory Retrieval (aSTEP, group 12)

Task 5: given a query trajectory, retrieve the Top-K most similar historical trajectories,
represented as sequences of road segments.

## Quick start (Docker)

The only requirement is [Docker Desktop](https://www.docker.com/products/docker-desktop/)
(Windows, macOS or Linux). From the repository root:

```bash
docker compose up -d
```

This starts PostgreSQL only. It creates an empty database and applies the schemas in `db/init/`:

| Service | What it does | Where |
|---|---|---|
| `db` | PostgreSQL 17 + PostGIS | `localhost:5433` (user `str`, password `str`, database `trajectories`) |

Start the API separately when needed: `docker compose --profile api up -d --build api`.

```bash
docker compose ps                  # status
docker compose --profile api logs -f api  # follow the API log
docker compose down                # stop (data is kept)
docker compose down -v             # stop and delete the database (next up re-creates the empty schemas)
docker compose up -d --build       # rebuild after code changes
```

Look at the data with any PostgreSQL client (e.g. the VS Code "PostgreSQL" extension, DBeaver or pgAdmin)
on `localhost:5433`, or directly in the container:

```bash
docker compose exec db psql -U str -d trajectories
```

The database uses port 5433 rather than 5432 so it does not clash with a locally installed PostgreSQL.
If 5433 or 5038 is taken, copy `.env.example` to `.env` and change `DB_PORT` / `API_PORT`.

## Structure

```
src/
  TrajectoryRetrieval.Api/    ASP.NET Core web API
  TrajectoryRetrieval.Core/   Domain models and interfaces (no dependencies)
tests/
  TrajectoryRetrieval.Tests/  xUnit tests
python/
  scripts/                    Data bundle pack/import tools
db/init/                      PostGIS, TSMini and TrajCL schemas (applied to a new database)
docker-compose.yml            PostgreSQL and optional API/data-import jobs
```

Everything other groups are responsible for sits behind an interface in `Core/Abstractions`
(`ITrajectorySource`, `IMapMatcher`, `ITrajectorySimilarity`, `ITravelTimeEstimator`).
We build our own simple implementations first and swap in the other groups' solutions later
through dependency injection in `Program.cs`. `ITrajectoryRetriever` is our own core logic.

## Data

Database initialization installs PostGIS and creates only the `tsmini` and `trajcl` schemas.
There are no Harbin road-network or travel-time tables in the project database.

### Beijing data for TSMini and TrajCL

Start PostgreSQL, then download a dataset zip and import it. Import verifies the zip in a temporary folder,
loads it into PostgreSQL, and removes the temporary files. It does not populate `TSMini/data` or `TrajCL/data`.

```powershell
docker compose up -d db
python scripts/share_data.py import "C:\path\to\beijing_20200_<date>_<name>.zip"
```

TSMini trajectories and Fréchet ground truth go into `tsmini.dataset`, `tsmini.trajectory`, and
`tsmini.ground_truth`; TrajCL's bundled test inputs are split into rows in `trajcl.test_trajectory`. Beijing model
commands use PostgreSQL as their input source. The query command does not need local dataset or checkpoint files.

Beijing model loaders select PostgreSQL automatically. The connection defaults to localhost port 5433, database
`trajectories`, user/password `str`. Override these in PowerShell only if your Docker database uses different settings:

```powershell
$env:PGHOST = "localhost"
$env:PGPORT = "5433"
$env:PGUSER = "str"
$env:PGPASSWORD = "str"
$env:PGDATABASE = "trajectories"
$env:TRAINING_DATA_SOURCE = "db"
```

Run TrajCL's full pipeline with `python TrajCL/train_pipeline.py --dataset beijing` from the repository
root. It checks PostgreSQL for completed stages and saves newly generated model artifacts to the database.
TSMini training can be run with `python train_trajsimi.py --dataset beijing` from `TSMini/`.

### Sharing data and models with the team

Datasets and trained models are too big for git, so they are shared as one zip per dataset in the team's
Google Drive folder. `scripts/share_data.py` only needs plain Python (no venv):

```bash
python scripts/share_data.py pack --prefix beijing_20200   # -> shared/beijing_20200_<date>_<name>.zip, upload it
python scripts/share_data.py list shared/<file>.zip         # who packed it, which models are inside
python scripts/share_data.py import shared/<file>.zip       # load into your local database; no local data copies
```

Workflow: download the newest zip → import → train your part → `pack` → upload. The zip holds the trajectories,
ground truth, TrajCL's node2vec cell embeddings, all trained models (`.pt`), embeddings and logs, each with a
checksum. `import` leaves no local data copies; use `unpack` only when a workflow explicitly requires local files.
Model file names are fixed, so **agree who trains which model** before starting, otherwise two people's checkpoints
of the same model conflict.

### Sharing an exact database snapshot

The full PostgreSQL snapshot includes the Beijing training inputs, trained model files, embeddings, test inputs,
and logs. Create and checksum a fresh compressed snapshot with:

```powershell
python scripts/database_dump.py dump shared/beijing_20200_20261117.dump
```

Share both the `.dump` and adjacent `.sha256` file. A teammate starts PostgreSQL, puts both files in `shared/`,
then restores the snapshot with:

```powershell
docker compose up -d db
python scripts/database_dump.py restore shared/beijing_20200_20261117.dump
```

Restore replaces the contents of that local `trajectories` database. The script checks the checksum and streams the
archive into PostgreSQL without unpacking it to a data folder.

### TrajCL (schema `trajcl`)

[db/init/04_trajcl.sql](db/init/04_trajcl.sql): TrajCL's trained models, node2vec cell embeddings, logs and
per-trajectory embeddings, saved and restored with `docker compose run --rm --build trajcl-import` (see
[TrajCL/readme.md](TrajCL/readme.md)). TrajCL uses the same trajectories and ground truth as TSMini, so those
are only in `tsmini`; `trajcl.embedding.idx` is `tsmini.trajectory.idx`.

### TSMini (schema `tsmini`)

Inputs and outputs of [TSMini](TSMini/) in [db/init/03_tsmini.sql](db/init/03_tsmini.sql).
The zip import command above loads Beijing trajectories, model outputs, and logs into PostgreSQL without leaving local
copies. The low-level TSMini importer can still be run directly, but expects files under `TSMini/data/` and `TSMini/exp/`.
For teammates, use the ZIP import command above.

| Table | Content |
|---|---|
| `dataset`, `trajectory` | Input: Beijing trajectories from `data/beijing_20200.h5`, with train/eval/test/unused splits |
| `ground_truth_set`, `ground_truth` | Input: Fréchet distances TSMini trains on, split into train/eval/test rows |
| `embedding` | Output: the 128-number embedding of every trajectory |
| `training_log` | Output: `exp/log/*.log` |
| `file` | Output: the trained model (`exp/snapshots/*.pt`) and plots, as `bytea` |

Run from PowerShell in the repository root after `docker compose up -d db`. This prints the embedding rank, L1 distance, exact route Frechet distance in metres, whether each match is in the true exact top 10, and the top-50 reranking recall. Exact Frechet is calculated against all Beijing trajectories; the database is only read:

```powershell
python scripts/query_models.py --model tsmini --idx 210
python scripts/query_models.py --model trajcl --idx 210
```

Replace `210` with a trajectory index from `0` to `49933`. Use `--k 5` to show 5 embedding matches or `--rerank 100` to rerank the nearest 100 embedding candidates. The query is by an existing trajectory index, not a new raw GPS route.

TSMini inputs use `tsmini.dataset`, `tsmini.trajectory`, and `tsmini.ground_truth`. TrajCL's robustness inputs
use `trajcl.test_trajectory`; cell-space data and model files
are stored in `trajcl.file`. Set `PGHOST=localhost` and `TRAINING_DATA_SOURCE=db` to load model inputs from PostgreSQL.

## C#

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download).

```bash
dotnet build
dotnet test
dotnet run --project src/TrajectoryRetrieval.Api   # http://localhost:5038/health
```
You can also instead of `dotnet run --project src/TrajectoryRetrieval.Api` do the following: 
```bash 
cd src
cd TrajectoryRetrieval.Api
dotnet run
```

When run outside Docker, the API connects to the Docker database on `localhost:5433`
(`ConnectionStrings:Default` in `appsettings.json`). Stop the `api` container first
(`docker compose stop api`) if you want to use port 5038 for `dotnet run`.

## Python

Requires Python 3.10+. Only needed for writing and running scripts yourself; the data import runs in Docker.

macOS / Linux:

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Windows (PowerShell):

```powershell
cd python
py -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
pytest
```

Scripts connect to the Docker database on `localhost:5433` by default
(override with the `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` environment variables).
