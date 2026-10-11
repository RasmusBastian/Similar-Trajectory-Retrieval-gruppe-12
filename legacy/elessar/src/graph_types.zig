pub const EdgeConnection = struct {
    edge_id: u64,
    vertex_start_id: u64,
    vertex_end_id: u64,
};

pub const Node = struct {
    id: u64,
    lat: f64,
    lon: f64,
};

pub const Edge = struct {
    edge_id: u64,
    to: u64,
    time: f32,
};

pub const EdgeInit = struct {
    from: u64,
    to: u64,
    edge: Edge,
};
