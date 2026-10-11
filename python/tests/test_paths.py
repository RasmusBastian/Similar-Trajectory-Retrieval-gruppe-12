from trajectory_algorithms import paths


def test_road_network_files_exist():
    assert paths.VERTEX_CSV.is_file()
    assert paths.EDGE_CONNECTIONS_CSV.is_file()
    assert paths.TRAVEL_TIME_DIR.is_dir()
