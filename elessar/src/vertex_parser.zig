const std = @import("std");
const types = @import("graph_types.zig");

pub fn parseVertexCsv(allocator: std.mem.Allocator, file: std.fs.File) ![]types.Node {
    const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(content);

    var list = std.ArrayList(types.Node).empty;
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

        const vertex = types.Node{
            .id = node_id,
            .lat = latitude,
            .lon = longitude,
        };

        try list.append(allocator, vertex);
    }

    return list.toOwnedSlice(allocator);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const alloc: std.mem.Allocator = gpa.allocator();
    const file = try std.fs.cwd().openFile("data/vertex.csv", .{ .mode = .read_only });
    defer file.close();

    const vertices = try parseVertexCsv(alloc, file);

    const first_vertex = vertices.items[0];
    std.debug.print("First vertex:\n  id: {d}\n  lat: {d}\n  lon: {d}\n", .{ first_vertex.id, first_vertex.lat, first_vertex.lon });
}
