"""Paths to the input data, which lives in the legacy folders."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = REPO_ROOT / "legacy"

VERTEX_CSV = LEGACY_DIR / "telchar" / "output" / "vertex.csv"
EDGE_CONNECTIONS_CSV = LEGACY_DIR / "telchar" / "output" / "edge_connections.csv"
# Cleaned per-day travel time matrices: edge_data_day{3..7}_cleaned.csv
TRAVEL_TIME_DIR = LEGACY_DIR / "postprocessing" / "output"
