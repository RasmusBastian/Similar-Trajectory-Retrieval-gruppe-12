# Telchar - GPS Trajectory Map Matching

Map-matching GPS trajectory data to road networks using FMM (Fast Map Matching) with STMatch algorithm.

## Project Structure

```
telchar/
├── input/           # GPS trip data (semicolon-separated CSV)
├── output/          # Processed results (edge traversal times, matched paths)
├── tmp/             # Generated files (OSM shapefiles, cache)
├── main.py          # Main processing pipeline
├── dl.py            # Download OSM road network
└── avg.py           # Average results across days
```

## Prerequisites

- Python 3.10+
- FMM library (pre-built binaries included)

### Python Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

All scripts must be run with `LD_LIBRARY_PATH=.` to load the FMM shared library:

### 1. Prepare Input Data

Place GPS trajectory CSV files in `input/`. Files must be semicolon-separated with columns:
- `trip_id` - unique identifier for each trip
- `latitude` - GPS latitude
- `longitude` - GPS longitude
- `timestamp` - Unix timestamp

### 2. Download Road Network

Download OpenStreetMap road network for the area covered by your GPS data:

```bash
LD_LIBRARY_PATH=. python dl.py
```

This creates shapefiles in `tmp/osm/harbin/`.

### 3. Run Map Matching

Process GPS trajectories and generate edge traversal data:

```bash
LD_LIBRARY_PATH=. python main.py
```

Outputs:
- `output/vertex.csv` - road network vertices
- `output/edge_connections.csv` - edge connectivity
- `output/matched_data_day*.txt` - raw matching results
- `output/edge_data_day*.csv` - time-edge traversal matrices

### 4. Average Results (Optional)

Compute average traversal times across multiple days:

```bash
LD_LIBRARY_PATH=. python avg.py
```

Outputs `output/edge_data_averaged.csv`.

## Output Format

### edge_data_day*.csv

Time-edge matrix with 5-minute intervals:
- Rows: time slots (00:00, 00:05, ..., 23:55)
- Columns: edge traversal times in seconds
- Value `-1`: no data for that time slot

### edge_connections.csv

Edge topology:
- `edge_id` - unique edge identifier
- `vertex_start_id` - source node
- `vertex_end_id` - target node

### vertex.csv

Node coordinates:
- `node_id` - unique node identifier
- `longitude`, `latitude` - WGS84 coordinates
