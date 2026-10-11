# Legacy

Code and data from the previous aSTEP group (travel time estimation, task 3), forked from
[astep-aau/time-travel-estimation-cs-25-sw-5-11](https://github.com/astep-aau/time-travel-estimation-cs-25-sw-5-11).
We do not develop on this code, but we use its data.

| Folder | What it is |
|---|---|
| `telchar/` | Map matching of Harbin GPS trips (Python + FMM, Linux x86_64 only) |
| `postprocessing/` | Cleans the travel time matrices (Zig) |
| `training/` | PCA+LSTM travel time model, exported to ONNX (PyTorch) |
| `elessar/` | Routing web API: A* with LSTM-predicted edge times (Zig) |

## Data we use

| Data | Path |
|---|---|
| Road network vertices | `telchar/output/vertex.csv` |
| Road network edges | `telchar/output/edge_connections.csv` |
| Cleaned travel times per edge per 5 min, days 3–7 | `postprocessing/output/edge_data_day*_cleaned.csv` |

`vertex.csv` and `edge_connections.csv` are identical in all four folders.

## Running the old server

```bash
cd legacy/elessar
./zig/zig build run   # see elessar/README.md for installing Zig; on macOS also onnxruntime-osx-arm64
```
