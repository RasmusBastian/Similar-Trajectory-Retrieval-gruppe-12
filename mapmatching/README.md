# Map matching of Beijing data

Runs last year's map matching ([legacy/telchar](../legacy/telchar), FMM with STMatch) on the Beijing data in the
shared aSTEP database and stores the result in our own database (`bj_matched_trajectory`, see
[db/init/02_beijing_map_matching.sql](../db/init/02_beijing_map_matching.sql)).

## The data has no GPS traces

Each Beijing trip has only two GPS points (start and end) and the list of road segments it drove on
(`road_points`, already map-matched by the [DRPK](https://github.com/derekwtian/DRPK) authors). To still test
the map matching, [match_beijing.py](match_beijing.py) simulates a GPS trace: a point every 15 s along the
trip's road segments, with 10 m Gaussian noise. FMM matches the trace, and the result is compared with
`road_points`:

- **precision**: share of the matched edges that the trip actually drove on
- **recall**: share of the trip's edges that FMM found

## Two parameter sets

STMatch parameters are in the units of the road network, which is in degrees here.

| Config | k | radius | gps_error | vmax | |
|---|---|---|---|---|---|
| `legacy` | 8 | 300 | 50 | 30 | FMM defaults, as in legacy/telchar/main.py (meant for meters) |
| `degrees` | 8 | 0.003 | 0.0005 | 0.0003 | the same values converted to degrees (~300 m, 50 m, 30 m/s) |

## Running

1. Start the shared database and import the data with [astep-aau/data](https://github.com/astep-aau/data)
   (`docker compose up -d --build` there; it listens on localhost:5432).
2. From this repository:

   ```bash
   docker compose run --rm --build mapmatching --limit 1000
   ```

   Options: `--dataset traj_test`, `--limit`, `--interval` (seconds between GPS points), `--noise` (meters),
   `--configs legacy,degrees`, `--seed`. Another shared database can be used by setting `SHARED_DATABASE_URL`.

FMM is built from source in the [Dockerfile](Dockerfile), because the prebuilt binaries in legacy/telchar need the
exact Arch Linux libraries they were built against.
