from datetime import timedelta
from typing import Dict, List

import geopandas as gpd
import pandas as pd
from fmm import STMATCH, GPSConfig, Network, NetworkGraph, ResultConfig, STMATCHConfig


def get_edge_lengths_from_shapefile(shapefile_path: str) -> Dict[int, float]:
    """Load edge lengths from shapefile and return as dictionary.

    Args:
        shapefile_path: Path to the shapefile containing edge geometries

    Returns:
        Dictionary mapping edge_id to length in meters
    """
    print("Loading edge lengths from shapefile...")
    edges_gdf: gpd.GeoDataFrame = gpd.read_file(shapefile_path)

    # Project to metric CRS if needed
    if edges_gdf.crs.is_geographic:
        edges_gdf = edges_gdf.to_crs("EPSG:32651")  # UTM for Harbin

    edge_lengths: Dict[int, float] = {}
    for idx, row in edges_gdf.iterrows():
        edge_id: int = row["fid"]
        edge_lengths[edge_id] = row.geometry.length

    print(f"Loaded {len(edge_lengths)} edge lengths")
    return edge_lengths


def parse_stmatch_results(
    result_file: str, gps_file: str, edge_lengths: Dict[int, float]
) -> pd.DataFrame:
    """Parse STMatch results and calculate traversal information.

    Args:
        result_file: Path to STMatch result file
        gps_file: Path to GPS data file
        edge_lengths: Dictionary of edge lengths

    Returns:
        DataFrame with traversal information (timestamp, edge_id, speed, time)
    """
    print("Parsing STMatch results...")
    result_df: pd.DataFrame = pd.read_csv(result_file, sep=";")
    gps_df: pd.DataFrame = pd.read_csv(gps_file, sep=";")

    result_df["edges"] = result_df["cpath"].apply(
        lambda x: [int(e) for e in str(x).split(",")]
        if pd.notna(x) and str(x) != ""
        else []
    )

    traversals: List[Dict[str, float]] = []

    for idx, row in result_df.iterrows():
        trip_id: int = row["id"]
        edges: List[int] = row["edges"]

        if not edges or len(edges) == 0:
            continue

        # Get GPS points for this trip
        trip_gps: pd.DataFrame = (
            gps_df[gps_df["trip_id"] == trip_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(trip_gps) < 2:
            continue

        # Calculate traversal info for each edge
        total_time: float = (
            trip_gps["timestamp"].iloc[-1] - trip_gps["timestamp"].iloc[0]
        )
        total_length: float = sum([edge_lengths.get(edge_id, 100) for edge_id in edges])

        if total_time <= 0 or total_length <= 0:
            continue

        # Distribute time across edges proportionally
        cumulative_time: float = trip_gps["timestamp"].iloc[0]

        for edge_id in edges:
            edge_length: float = edge_lengths.get(edge_id, 100)
            edge_time: float = (edge_length / total_length) * total_time

            # Calculate speed
            if edge_time > 0:
                speed_ms: float = edge_length / edge_time
                speed_kmh: float = speed_ms * 3.6
            else:
                speed_kmh: float = 0

            traversals.append(
                {
                    "timestamp": cumulative_time,
                    "edge_id": edge_id,
                    "speed_kmh": round(speed_kmh, 2),
                    "traversal_time_seconds": round(edge_time, 2),
                }
            )

            cumulative_time += edge_time

    return pd.DataFrame(traversals)


def create_time_edge_matrix(
    traversals_df: pd.DataFrame, edge_lengths: Dict[int,float], time_interval_minutes: int = 5
) -> pd.DataFrame:
    """Create time-edge matrix with traversal times.

    Args:
        traversals_df: DataFrame containing traversal data
        time_interval_minutes: Time interval for aggregation in minutes

    Returns:
        DataFrame with time slots as rows and edges as columns
    """
    print("Creating time edge matrix")
    # Convert Unix timestamps to datetime
    traversals_df["datetime"] = pd.to_datetime(traversals_df["timestamp"], unit="s")

    # Find start and end of day
    start_of_day: pd.Timestamp = (
        traversals_df["datetime"]
        .min()
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )
    end_of_day: pd.Timestamp = start_of_day + timedelta(days=1)

    time_slots: pd.DatetimeIndex = pd.date_range(
        start=start_of_day, end=end_of_day, freq=f"{time_interval_minutes}min"
    )[:-1]

    # Assign each traversal to a time slot
    traversals_df["time_slot"] = pd.cut(
        traversals_df["datetime"],
        bins=pd.date_range(
            start=start_of_day, end=end_of_day, freq=f"{time_interval_minutes}min"
        ),
        labels=time_slots,
        include_lowest=True,
    )

    all_edges: List[int] = sorted(edge_lengths.keys())

    # Aggregate: for each (time_slot, edge), calculate mean traversal time
    aggregated: pd.DataFrame = (
        traversals_df.groupby(["time_slot", "edge_id"])["traversal_time_seconds"]
        .mean()
        .reset_index()
    )

    # Pivot to create matrix: rows=time_slots, columns=edges
    matrix: pd.DataFrame = aggregated.pivot(
        index="time_slot", columns="edge_id", values="traversal_time_seconds"
    )

    # Reindex to include all time slots and all edges
    matrix = matrix.reindex(index=time_slots, columns=all_edges)

    # Fill missing values with -1 (no data for that time slot)
    matrix = matrix.fillna(-1)

    matrix.index = matrix.index.strftime("%H:%M")
    matrix.columns = [f"edge{edge_id}_traversal_time_sec" for edge_id in matrix.columns]
    matrix = matrix.reset_index()
    matrix = matrix.rename(columns={"index": "time_slot"})

    return matrix


def generate_vertex_csv(
    shapefile_path: str, output_file: str = "vertex.csv"
) -> pd.DataFrame:
    """Generate vertex CSV from shapefile.

    Args:
        shapefile_path: Path to the shapefile
        output_file: Path to save the vertex CSV

    Returns:
        DataFrame containing vertex information (node_id, longitude, latitude)
    """
    print("Generating vertex.csv...")
    edges_gdf: gpd.GeoDataFrame = gpd.read_file(shapefile_path)

    # Ensure we're in lat/lon
    if not edges_gdf.crs.is_geographic:
        edges_gdf = edges_gdf.to_crs("EPSG:4326")

    vertices: Dict[int, Dict[str, float]] = {}

    for idx, row in edges_gdf.iterrows():
        u: int = row["u"]
        v: int = row["v"]
        geom = row.geometry

        start_point = geom.coords[0]
        end_point = geom.coords[-1]

        if u not in vertices:
            vertices[u] = {"longitude": start_point[0], "latitude": start_point[1]}
        if v not in vertices:
            vertices[v] = {"longitude": end_point[0], "latitude": end_point[1]}

    # Create DataFrame
    vertex_df: pd.DataFrame = pd.DataFrame(
        [
            {
                "node_id": node_id,
                "longitude": coords["longitude"],
                "latitude": coords["latitude"],
            }
            for node_id, coords in vertices.items()
        ]
    )

    vertex_df = vertex_df.sort_values("node_id").reset_index(drop=True)
    vertex_df.to_csv(output_file, index=False)

    print(f"Saved {len(vertex_df)} vertices to {output_file}")
    return vertex_df


def generate_edge_connections_csv(
    shapefile_path: str, output_file: str = "edge_connections.csv"
) -> pd.DataFrame:
    """Generate edge connections CSV from shapefile.

    Args:
        shapefile_path: Path to the shapefile
        output_file: Path to save the edge connections CSV

    Returns:
        DataFrame containing edge connection information
    """
    print("Generating edge_connections.csv...")
    edges_gdf: gpd.GeoDataFrame = gpd.read_file(shapefile_path)

    edge_connections: List[Dict[str, int]] = []

    for idx, row in edges_gdf.iterrows():
        edge_connections.append(
            {
                "edge_id": row["fid"],
                "vertex_start_id": row["u"],
                "vertex_end_id": row["v"],
            }
        )

    edge_conn_df: pd.DataFrame = pd.DataFrame(edge_connections)
    edge_conn_df = edge_conn_df.sort_values("edge_id").reset_index(drop=True)
    edge_conn_df.to_csv(output_file, index=False)

    print(f"Saved {len(edge_conn_df)} edge connections to {output_file}")
    return edge_conn_df


def process_single_day(
    gps_file: str,
    network_file: str,
    result_file: str,
) -> pd.DataFrame:
    """Process a single day of GPS data.

    Args:
        gps_file: Path to GPS data file
        network_file: Path to network shapefile
        result_file: Path to save matching results

    Returns:
        DataFrame containing the time-edge matrix
    """
    print(f"\n=== Processing file: {gps_file} ===")

    network: Network = Network(network_file, "fid", "u", "v")
    graph: NetworkGraph = NetworkGraph(network)
    model: STMATCH = STMATCH(network, graph)
    stmatch_config: STMATCHConfig = STMATCHConfig()

    input_config: GPSConfig = GPSConfig()
    input_config.file = gps_file
    input_config.id = "trip_id"
    input_config.x = "longitude"
    input_config.y = "latitude"
    input_config.timestamp = "timestamp"
    input_config.gps_point = True

    result_config: ResultConfig = ResultConfig()
    result_config.file = result_file

    model.match_gps_file(input_config, result_config, stmatch_config)

    edge_lengths: Dict[int, float] = get_edge_lengths_from_shapefile(network_file)
    traversals_df: pd.DataFrame = parse_stmatch_results(
        result_file, gps_file, edge_lengths
    )
    matrix: pd.DataFrame = create_time_edge_matrix(
        traversals_df, time_interval_minutes=5,
        edge_lengths=edge_lengths
    )

    return matrix


if __name__ == "__main__":
    gps_files: List[str] = [
        "./input/short.csv",
        "./input/trips_150103.csv",
        "./input/trips_150104.csv",
        "./input/trips_150105.csv",
        "./input/trips_150106.csv",
        "./input/trips_150107.csv",
    ]
    network_file: str = "./tmp/osm/harbin/edges.shp"

    vertex_df: pd.DataFrame = generate_vertex_csv(
        network_file, "./output/vertex.csv"
    )
    edge_conn_df: pd.DataFrame = generate_edge_connections_csv(
        network_file, "./output/edge_connections.csv"
    )

    for i, gps_file in enumerate(gps_files, start=3):
        result_file: str = f"./output/matched_data_day{i}.txt"
        edge_data_file: str = f"./output/edge_data_day{i}.csv"
        matrix: pd.DataFrame = process_single_day(gps_file, network_file, result_file)
        matrix.to_csv(edge_data_file, index=False)
        print(f"Matrix saved to {edge_data_file}")
        print(f"Shape: {matrix.shape}")
        print(f"Non-empty cells: {(matrix.iloc[:, 1:] != -1).sum().sum()}")
