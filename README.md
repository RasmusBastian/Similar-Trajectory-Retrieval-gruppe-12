# Similar Trajectory Retrieval (aSTEP, group 12)

Task 5: given a query trajectory, retrieve the Top-K most similar historical trajectories,
represented as sequences of road segments.

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
legacy/                       Previous group's code and data (see legacy/README.md)
```

Everything other groups are responsible for sits behind an interface in `Core/Abstractions`
(`ITrajectorySource`, `IMapMatcher`, `ITrajectorySimilarity`, `ITravelTimeEstimator`).
We build our own simple implementations first and swap in the other groups' solutions later
through dependency injection in `Program.cs`. `ITrajectoryRetriever` is our own core logic.

## C#

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download).

```bash
dotnet build
dotnet test
dotnet run --project src/TrajectoryRetrieval.Api   # http://localhost:5038/health
```
You can also instead of `dotnet run --project src/TrajectoryRetrieval.Api` 
do `cd src`
Then `cd TrajectoryRetrieval.Api`
Then `dotnet run`

## Python

Requires Python 3.10+.

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data

Input data is read directly from `legacy/` (paths in `src/TrajectoryRetrieval.Api/appsettings.json`
and `python/trajectory_algorithms/paths.py`).
