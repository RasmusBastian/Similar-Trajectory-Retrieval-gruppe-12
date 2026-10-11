const std = @import("std");
const types = @import("graph_types.zig");
const node_parse = @import("vertex_parser.zig");
const edge_parse = @import("edge_connections_parser.zig");
const Query = @import("lookup_table.zig").Query;

// 2x to
// time er konstant f64
// ellers NODE = return fra vertex_parser
// edge init (-edge time) = return fra edge_csv_parser
// time array kan faas fra edge_connections_parser
//TODO vertex_parser og astar har forskellig struct af nodes
//TODO lav time i Edge til array, lav get_time = time[input]
//TODO add en form for start tid til astar som bliver incremented for hver edge traversed?

pub const Graph = struct {
    nodes: []types.Node,
    adj: [][]types.Edge,
    id_to_index: std.AutoHashMap(u64, usize),

    pub fn init(alloc: std.mem.Allocator, nodes: []types.Node, edges: []const types.EdgeInit) !Graph {
        // Create mapping from node ID to array index
        var id_to_index = std.AutoHashMap(u64, usize).init(alloc);
        errdefer id_to_index.deinit();

        for (nodes, 0..) |node, i| {
            try id_to_index.put(node.id, i);
        }

        const adj = try alloc.alloc([]types.Edge, nodes.len);
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
            list.* = try alloc.alloc(types.Edge, count);
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

const SearchNode = struct {
    id: u64,
    f: f64,
    time_offset: f64,
};

pub const Journey = struct {
    stops: []u64,
    estimated_traversal_time_seconds: usize,
    length_m: f64
};

fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) f64 {
    const R = 6371000.0;
    const phi1 = lat1 * std.math.pi / 180.0;
    const phi2 = lat2 * std.math.pi / 180.0;
    const dphi = (lat2 - lat1) * std.math.pi / 180.0;
    const dlam = (lon2 - lon1) * std.math.pi / 180.0;

    const a = @sin(dphi / 2.0) * @sin(dphi / 2.0) +
        @cos(phi1) * @cos(phi2) * @sin(dlam / 2.0) * @sin(dlam / 2.0);
    const c = 2.0 * std.math.atan2(@sqrt(a), @sqrt(1.0 - a));

    return @floatCast(R * c);
}

fn lessThan(_: void, a: SearchNode, b: SearchNode) std.math.Order {
    return std.math.order(a.f, b.f);
}

pub fn aStar(
    alloc: std.mem.Allocator,
    graph: Graph,
    start_id: u64,
    goal_id: u64,
    lookup_table: *const std.AutoHashMap(Query, f32),
    timestep_seconds: f64,
) !Journey {
    if (start_id == goal_id) {
        const stops = try alloc.alloc(u64, 1);
        stops[0] = start_id;
        return Journey{
            .stops = stops,
            .estimated_traversal_time_seconds = 0,
            .length_m = 0
        };
    }

    var open = std.PriorityQueue(SearchNode, void, lessThan).init(alloc, {});
    defer open.deinit();

    var openSet = std.AutoHashMap(u64, void).init(alloc);
    defer openSet.deinit();

    var cameFrom = std.AutoHashMap(u64, u64).init(alloc);
    defer cameFrom.deinit();

    var gScore = std.AutoHashMap(u64, f64).init(alloc);
    defer gScore.deinit();

    try gScore.put(start_id, 0.0);
    try open.add(SearchNode{ .id = start_id, .f = 0.0, .time_offset = 0.0 });
    try openSet.put(start_id, {});

    // Cache goal node coordinates for heuristic calculation
    const goal_idx = graph.id_to_index.get(goal_id) orelse return error.NodeNotFound;
    const goal_node = graph.nodes[goal_idx];
    const goal_lat = goal_node.lat;
    const goal_lon = goal_node.lon;

    while (open.count() > 0) {
        const current = open.remove();
        _ = openSet.remove(current.id);

        if (current.id == goal_id) {
            var path: std.ArrayList(u64) = .empty;
            defer path.deinit(alloc);

            var curr = goal_id;
            try path.append(alloc, curr);
            while (cameFrom.get(curr)) |prev| {
                curr = prev;
                try path.append(alloc, curr);
            }
            std.mem.reverse(u64, path.items);

            // Calculate total travel time
            const total_time_seconds = gScore.get(goal_id).?;

            var total_distance: f64 = 0.0;
            for (0..path.items.len - 1) |i| {
                const from_idx = graph.id_to_index.get(path.items[i]) orelse return error.NodeNotFound;
                const to_idx = graph.id_to_index.get(path.items[i + 1]) orelse return error.NodeNotFound;

                const from_node = graph.nodes[from_idx];
                const to_node = graph.nodes[to_idx];

                total_distance += haversine(from_node.lat, from_node.lon, to_node.lat, to_node.lon);
            }

            // Debug: print the edges in the final path
            std.debug.print("\n=== FINAL PATH ===\n", .{});
            std.debug.print("Total time: {d:.2}s, Total distance: {d:.2}m\n", .{ total_time_seconds, total_distance });
            for (0..path.items.len - 1) |i| {
                const from_idx = graph.id_to_index.get(path.items[i]) orelse continue;
                const to_id = path.items[i + 1];
                // Find the edge
                for (graph.adj[from_idx]) |edge| {
                    if (edge.to == to_id) {
                        const edge_time = lookup_table.get(.{ .edge_id = edge.edge_id, .time_offset = 0 }) orelse edge.time;
                        std.debug.print("  Step {}: edge {} -> {d:.2}s\n", .{ i, edge.edge_id, edge_time });
                        break;
                    }
                }
            }
            std.debug.print("==================\n\n", .{});

            return Journey{
                .stops = try path.toOwnedSlice(alloc),
                .estimated_traversal_time_seconds = @intFromFloat(@round(total_time_seconds)),
                .length_m = total_distance
            };
        }

        const currentG = gScore.get(current.id).?;

        // Calculate time_offset index using integer division
        const current_time_offset: u64 = @intFromFloat(currentG / timestep_seconds);

        // Convert current node ID to index
        const current_idx = graph.id_to_index.get(current.id) orelse return error.NodeNotFound;

        var debug_edge_count: usize = 0;
        for (graph.adj[current_idx]) |edge| {
            const neighbor_id = edge.to;

            // Query the lookup table for the edge travel time (in seconds)
            const query = Query{ .edge_id = edge.edge_id, .time_offset = current_time_offset };
            const lookup_result = lookup_table.get(query);
            const travel_time = @max(0.0, lookup_result orelse edge.time);

            // Debug first few edges
            if (debug_edge_count < 3) {
                std.debug.print("Edge {}: lookup({}, {}) = {?d:.2}, fallback={d:.2}, used={d:.2}\n", .{
                    edge.edge_id, edge.edge_id, current_time_offset, lookup_result, edge.time, travel_time
                });
                debug_edge_count += 1;
            }

            const tentativeG = currentG + travel_time;
            const existingG = gScore.get(neighbor_id) orelse std.math.floatMax(f64);

            if (tentativeG < existingG) {
                try cameFrom.put(neighbor_id, current.id);
                try gScore.put(neighbor_id, tentativeG);

                if (!openSet.contains(neighbor_id)) {
                    // Get neighbor coordinates for heuristic
                    const neighbor_idx = graph.id_to_index.get(neighbor_id) orelse return error.NodeNotFound;
                    const neighbor_node = graph.nodes[neighbor_idx];

                    // Use cached goal coordinates
                    const h = haversine(neighbor_node.lat, neighbor_node.lon, goal_lat, goal_lon) / 30.0;
                    try open.add(SearchNode{ .id = neighbor_id, .f = tentativeG + h, .time_offset = tentativeG });
                    try openSet.put(neighbor_id, {});
                }
            }
        }
    }

    return error.PathNotFound;
}

pub fn generate_network_graph(node_file: []const u8, edge_file: []const u8, alloc: std.mem.allocator) !Graph {
    const nodes = node_parse.parseVertexCsv(alloc, node_file);
    const edges = edge_parse.parseEdgeConnectionsCsv(alloc, edge_file);

    const graph = try Graph.init(alloc, nodes, &edges);

    return graph;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const nodes = try alloc.alloc(types.Node, 5);
    nodes[0] = .{ .id = 0, .lat = 55.676, .lon = 12.568 };
    nodes[1] = .{ .id = 1, .lat = 55.678, .lon = 12.570 };
    nodes[2] = .{ .id = 2, .lat = 55.680, .lon = 12.575 };
    nodes[3] = .{ .id = 3, .lat = 55.682, .lon = 12.580 };
    nodes[4] = .{ .id = 4, .lat = 55.684, .lon = 12.585 };

    const edges = [_]types.EdgeInit{
        .{ .from = 0, .edge = .{ .edge_id = 0, .to = 1, .time = 30 } },
        .{ .from = 1, .edge = .{ .edge_id = 1, .to = 2, .time = 45 } },
        .{ .from = 2, .edge = .{ .edge_id = 2, .to = 3, .time = 50 } },
        .{ .from = 3, .edge = .{ .edge_id = 3, .to = 4, .time = 35 } },
        .{ .from = 1, .edge = .{ .edge_id = 4, .to = 3, .time = 80 } },
    };

    var graph = try Graph.init(alloc, nodes, &edges);
    defer graph.deinit(alloc);

    // Create an example lookup table
    var lookup_table = std.AutoHashMap(Query, f32).init(alloc);
    defer lookup_table.deinit();

    // Populate with some example values (in a real scenario, this comes from generateLookupTable)
    try lookup_table.put(.{ .edge_id = 0, .time_offset = 0 }, 28.0);
    try lookup_table.put(.{ .edge_id = 1, .time_offset = 0 }, 42.0);
    try lookup_table.put(.{ .edge_id = 2, .time_offset = 0 }, 48.0);

    // Each timestep represents 5 minutes (300 seconds)
    const timestep_seconds = 300.0;

    const journey = try aStar(alloc, graph, 0, 4, &lookup_table, timestep_seconds);
    defer journey.deinit(alloc);

    std.debug.print("Journey found: {d} stops\n", .{journey.stops.len});
    std.debug.print("Estimated traversal time: {d} seconds\n", .{journey.estimated_traversal_time_seconds});
    // for (journey.stops, 0..) |node_id, i| {
    //     const node = graph.nodes[node_id];
    //     std.debug.print("  {d}: Node {d} ({d:.6}, {d:.6})\n", .{ i, node_id, node.lat, node.lon });
    // }
}
