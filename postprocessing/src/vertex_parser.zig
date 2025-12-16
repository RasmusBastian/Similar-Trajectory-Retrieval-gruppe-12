const std = @import("std");
const graph = @import("graph.zig");

pub fn parseVertexCsv(allocator: std.mem.Allocator, file: std.fs.File) ![]graph.Node {
    const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(content);

    var list = std.ArrayList(graph.Node).empty;
    errdefer list.deinit(allocator);

    var lines = std.mem.splitScalar(u8, content, '\n');

    // Skip the first line with column names (node_id,longitude,latitude)
    _ = lines.next();

    while (lines.next()) |line| {
        if (line.len == 0) continue;

        var iter = std.mem.tokenizeScalar(u8, line, ',');

        // Parse node_id
        const node_id_str = iter.next() orelse continue;
        const node_id = try std.fmt.parseInt(u64, node_id_str, 10);

        // Parse longitude
        const longitude_str = iter.next() orelse continue;
        const longitude = try std.fmt.parseFloat(f64, longitude_str);

        // Parse latitude
        const latitude_str = iter.next() orelse continue;
        const latitude = try std.fmt.parseFloat(f64, latitude_str);

        const vertex = graph.Node{
            .id = node_id,
            .lat = latitude,
            .lon = longitude,
        };

        try list.append(allocator, vertex);
    }

    return list.toOwnedSlice(allocator);
}
