const std = @import("std");
const types = @import("graph_types.zig");

pub fn parseEdgeConnectionsCsv(allocator: std.mem.Allocator, file: std.fs.File) ![]types.EdgeInit {
    const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(content);

    var list = std.ArrayList(types.EdgeConnection).empty;
    errdefer list.deinit(allocator);

    var lines = std.mem.splitScalar(u8, content, '\n');

    // Skip the first line with column names (edge_id,vertex_start_id,vertex_end_id)
    _ = lines.next();

    while (lines.next()) |line| {
        if (line.len == 0) continue;

        var iter = std.mem.tokenizeScalar(u8, line, ',');

        // Parse edge_id
        const edge_id_str = iter.next() orelse continue;
        const edge_id = try std.fmt.parseInt(u64, edge_id_str, 10);

        // Parse vertex_start_id
        const vertex_start_id_str = iter.next() orelse continue;
        const vertex_start_id = try std.fmt.parseInt(u64, vertex_start_id_str, 10);

        // Parse vertex_end_id
        const vertex_end_id_str = iter.next() orelse continue;
        const vertex_end_id = try std.fmt.parseInt(u64, vertex_end_id_str, 10);

        const edge_connection = types.EdgeConnection{
            .edge_id = edge_id,
            .vertex_start_id = vertex_start_id,
            .vertex_end_id = vertex_end_id,
        };

        try list.append(allocator, edge_connection);
    }

    const parsedList = try edgeConnectionsToEdgeInits(allocator, list.items);
    list.deinit(allocator);

    return parsedList;
}

pub fn edgeConnectionsToEdgeInits(
    allocator: std.mem.Allocator,
    edge_conns: []const types.EdgeConnection,
) ![]types.EdgeInit {
    const n = edge_conns.len;
    const edge_inits = try allocator.alloc(types.EdgeInit, n);

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const conn = edge_conns[i];
        edge_inits[i] = types.EdgeInit{
            .from = conn.vertex_start_id,
            .to = conn.vertex_end_id,
            .edge = types.Edge{
                .edge_id = conn.edge_id,
                .to = conn.vertex_end_id,
                .time = 0.0,
            },
        };
    }

    return edge_inits;
}
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc: std.mem.Allocator = gpa.allocator();
    const file = try std.fs.cwd().openFile("data/edge_connections.csv", .{ .mode = .read_only });
    defer file.close();

    const edges = try parseEdgeConnectionsCsv(alloc, file);

    const first_edge = edges[0];
    std.debug.print("First edge:\n  from: {d}\n  to: {d}\n  time: {d}\n", .{ first_edge.from, first_edge.to, first_edge.edge.time });
}
