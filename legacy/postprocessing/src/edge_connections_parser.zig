const std = @import("std");
const graph = @import("graph.zig");

pub fn parseEdgeConnectionsCsv(allocator: std.mem.Allocator, file: std.fs.File) ![]graph.EdgeInit {
    const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(content);

    var list = std.ArrayList(graph.EdgeConnection).empty;
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

        const edge_connection = graph.EdgeConnection{
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
    edge_conns: []const graph.EdgeConnection,
) ![]graph.EdgeInit {
    const n = edge_conns.len;
    const edge_inits = try allocator.alloc(graph.EdgeInit, n);

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const conn = edge_conns[i];
        edge_inits[i] = graph.EdgeInit{
            .from = conn.vertex_start_id,
            .to = conn.vertex_end_id,
            .edge = graph.Edge{
                .edge_id = conn.edge_id,
                .to = conn.vertex_end_id,
                .time = 0.0,
            },
        };
    }

    return edge_inits;
}
