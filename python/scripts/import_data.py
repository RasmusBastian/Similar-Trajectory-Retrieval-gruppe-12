"""Import the road network and cleaned travel times from legacy/ into the database.

Runs automatically as the `importer` service in docker-compose.yml. Skips if the data is already there.
Can also be run on the host against the Docker database: python -m scripts.import_data
Connection settings come from the standard PG* environment variables (defaults match docker-compose.yml).
"""

import io
import os
import re

import pandas as pd
import psycopg

from trajectory_algorithms import paths


def connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5433"),
        user=os.environ.get("PGUSER", "str"),
        password=os.environ.get("PGPASSWORD", "str"),
        dbname=os.environ.get("PGDATABASE", "trajectories"),
    )


def copy_rows(cur: psycopg.Cursor, sql: str, buffer: io.StringIO) -> None:
    buffer.seek(0)
    with cur.copy(sql) as copy:
        while chunk := buffer.read(1 << 20):
            copy.write(chunk)


def import_road_network(cur: psycopg.Cursor) -> None:
    vertices = pd.read_csv(paths.VERTEX_CSV)
    buffer = io.StringIO()
    vertices[["node_id", "longitude", "latitude"]].to_csv(buffer, index=False, header=False)
    copy_rows(cur, "COPY vertex (node_id, longitude, latitude) FROM STDIN WITH (FORMAT csv)", buffer)
    print(f"  vertex: {len(vertices)} rows")

    edges = pd.read_csv(paths.EDGE_CONNECTIONS_CSV)
    cur.execute("CREATE TEMP TABLE edge_staging (edge_id INT, start_node BIGINT, end_node BIGINT) ON COMMIT DROP")
    buffer = io.StringIO()
    edges[["edge_id", "vertex_start_id", "vertex_end_id"]].to_csv(buffer, index=False, header=False)
    copy_rows(cur, "COPY edge_staging FROM STDIN WITH (FORMAT csv)", buffer)
    # Edge geometry is the straight line between its two vertices
    cur.execute("""
        INSERT INTO edge (edge_id, start_node, end_node, geom, length_m)
        SELECT s.edge_id, s.start_node, s.end_node,
               ST_MakeLine(a.geom, b.geom),
               ST_Length(ST_MakeLine(a.geom, b.geom)::geography)
        FROM edge_staging s
        JOIN vertex a ON a.node_id = s.start_node
        JOIN vertex b ON b.node_id = s.end_node
    """)
    print(f"  edge: {cur.rowcount} rows")


def import_travel_times(cur: psycopg.Cursor) -> None:
    files = sorted(paths.TRAVEL_TIME_DIR.glob("edge_data_day*_cleaned.csv"))
    for file in files:
        day = int(re.search(r"day(\d+)", file.name).group(1))
        matrix = pd.read_csv(file, index_col="time_slot")  # 288 time slots x one column per edge
        edge_ids = [int(re.match(r"edge(\d+)_", column).group(1)) for column in matrix.columns]

        buffer = io.StringIO()
        for edge_id, slots in zip(edge_ids, matrix.to_numpy().T):
            buffer.write(f"{day}\t{edge_id}\t{{{','.join(f'{s:g}' for s in slots)}}}\n")
        copy_rows(cur, "COPY edge_travel_time (day, edge_id, seconds) FROM STDIN", buffer)
        print(f"  edge_travel_time day {day}: {len(edge_ids)} rows")


def main() -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT EXISTS (SELECT 1 FROM edge), EXISTS (SELECT 1 FROM edge_travel_time)")
        has_edges, has_travel_times = cur.fetchone()
        if has_edges and has_travel_times:
            print("Data already imported, skipping.")
            return

        print("Importing data from legacy/ ...")
        cur.execute("TRUNCATE edge_travel_time, edge, vertex CASCADE")
        import_road_network(cur)
        import_travel_times(cur)
    print("Import done.")


if __name__ == "__main__":
    main()
