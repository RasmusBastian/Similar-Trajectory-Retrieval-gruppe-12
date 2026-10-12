import os
import time
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon


def save_graph_shapefile_directional(
    G: nx.MultiDiGraph, filepath: Optional[str] = None, encoding: str = "utf-8"
) -> None:
    """Save a graph as directional shapefiles with unique edge IDs.

    Args:
        G: NetworkX MultiDiGraph to save
        filepath: Directory path to save shapefiles. If None, uses default OSMnx folder
        encoding: Character encoding for shapefiles

    Returns:
        None
    """
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)

    filepath_nodes: str = os.path.join(filepath, "nodes.shp")
    filepath_edges: str = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes: gpd.GeoDataFrame
    gdf_edges: gpd.GeoDataFrame
    gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)

    # We need an unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype="int")

    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)


if __name__ == "__main__":
    print("osmnx version", ox.__version__)

    df: pd.DataFrame = pd.read_csv("input/trips_150103.csv", sep=";")

    y1: float = df["latitude"].max() + 0.01
    y2: float = df["latitude"].min() - 0.01
    x1: float = df["longitude"].max() + 0.01
    x2: float = df["longitude"].min() - 0.01

    boundary_polygon: Polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    G: nx.MultiDiGraph = ox.graph_from_polygon(boundary_polygon, network_type="drive")

    start_time: float = time.time()
    save_graph_shapefile_directional(G, filepath="./tmp/osm/harbin")
    print("--- %s seconds ---" % (time.time() - start_time))
