const std = @import("std");
const node_parse = @import("vertex_parser.zig");
const edge_parse = @import("edge_connections_parser.zig");

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

pub const Graph = struct {
    nodes: []Node,
    adj: [][]Edge,
    id_to_index: std.AutoHashMap(u64, usize),

    pub fn init(alloc: std.mem.Allocator, nodes: []Node, edges: []const EdgeInit) !Graph {
        // Create mapping from node ID to array index
        var id_to_index = std.AutoHashMap(u64, usize).init(alloc);
        errdefer id_to_index.deinit();

        for (nodes, 0..) |node, i| {
            try id_to_index.put(node.id, i);
        }

        const adj = try alloc.alloc([]Edge, nodes.len);
        for (adj) |*list| list.* = &.{};

        var counts = try alloc.alloc(usize, nodes.len);
        defer alloc.free(counts);
        @memset(counts, 0);

        // Use the mapping to convert vertex IDs to indices
        for (edges) |e| {
            const from_idx = id_to_index.get(e.from) orelse continue;
            counts[from_idx] += 1;
        }

        for (adj, counts) |*list, count| {
            list.* = try alloc.alloc(Edge, count);
        }

        @memset(counts, 0);
        for (edges) |e| {
            const from_idx = id_to_index.get(e.from) orelse continue;
            adj[from_idx][counts[from_idx]] = e.edge;
            counts[from_idx] += 1;
        }

        return .{ .nodes = nodes, .adj = adj, .id_to_index = id_to_index };
    }

    pub fn deinit(self: *Graph, alloc: std.mem.Allocator) void {
        for (self.adj) |list| alloc.free(list);
        alloc.free(self.adj);
        self.id_to_index.deinit();
    }
};
