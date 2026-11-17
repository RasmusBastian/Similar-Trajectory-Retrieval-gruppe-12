# Map matching experiment (separate from the TSMini/TrajCL database)

This experiment is not part of the current Docker Compose database workflow. The project database now initializes
only the TSMini and TrajCL schemas, and this experiment's output table is not created by Compose.

Runs FMM map matching (STMatch) on the Beijing data in the shared aSTEP database and can store results in a
separately provisioned `bj_matched_trajectory` table.

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
| `legacy` | 8 | 300 | 50 | 30 | Previous FMM defaults (meant for meters) |
| `degrees` | 8 | 0.003 | 0.0005 | 0.0003 | the same values converted to degrees (~300 m, 50 m, 30 m/s) |

## Running

This tool is kept as a separate experiment and is not run through this repository's Compose services.

FMM is built from source in the [Dockerfile](Dockerfile).
