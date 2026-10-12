# Similar Trajectory Retrieval (aSTEP, group 12)

Task 5: given a query trajectory, retrieve the Top-K most similar historical trajectories,
represented as sequences of road segments.

## Quick start (Docker)

The only requirement is [Docker Desktop](https://www.docker.com/products/docker-desktop/)
(Windows, macOS or Linux). From the repository root:

```bash
docker compose up -d
```

This starts three containers:

| Service | What it does | Where |
|---|---|---|
| `db` | PostgreSQL 17 + PostGIS | `localhost:5433` (user `str`, password `str`, database `trajectories`) |
| `importer` | Imports the road network and travel times from `legacy/` into `db`, then exits. Skipped if the data is already there | — |
| `api` | The C# API | <http://localhost:5038/health> |

The first start takes a few minutes (downloading images and building), later starts take seconds.

```bash
docker compose ps                  # status
docker compose logs -f api         # follow the API log
docker compose down                # stop (data is kept)
docker compose down -v             # stop and delete the database (next up re-creates and re-imports)
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
  trajectory_algorithms/      Algorithm-heavy code (similarity, embeddings, ...)
  scripts/                    One-off scripts (import, data generation, evaluation)
db/init/                      Database schema, applied when the database is first created
docker-compose.yml            Local environment: database, data import and API
legacy/                       Previous group's code and data (see legacy/README.md)
```

Everything other groups are responsible for sits behind an interface in `Core/Abstractions`
(`ITrajectorySource`, `IMapMatcher`, `ITrajectorySimilarity`, `ITravelTimeEstimator`).
We build our own simple implementations first and swap in the other groups' solutions later
through dependency injection in `Program.cs`. `ITrajectoryRetriever` is our own core logic.

## Data

The database schema is in [db/init/01_schema.sql](db/init/01_schema.sql).

| Table | Content |
|---|---|
| `vertex` | Road network nodes: `node_id`, `longitude`, `latitude`, `geom` (PostGIS point) |
| `edge` | Road segments: `edge_id`, `start_node`, `end_node`, `geom` (straight line), `length_m` |
| `edge_travel_time` | Cleaned travel times for days 3–7: one row per `(day, edge_id)`, `seconds[1..288]` per 5-minute slot (`seconds[1]` = 00:00–00:05) |
| `trajectory`, `trajectory_edge` | Trajectories as ordered edge sequences (empty for now) |

Edges 22748–22751 have no travel times in the source data.

Example: travel time of edge 0 at 08:00 on day 7 (slot = hour × 12 + minute / 5 + 1):

```sql
SELECT seconds[97] FROM edge_travel_time WHERE day = 7 AND edge_id = 0;
```

The source CSV files stay in `legacy/` (paths in `src/TrajectoryRetrieval.Api/appsettings.json`
and `python/trajectory_algorithms/paths.py`).

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
