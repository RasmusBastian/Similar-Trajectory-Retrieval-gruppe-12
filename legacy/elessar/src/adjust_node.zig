const std = @import("std");

pub const Vertex = struct {
    id: u64,
    lon: f64,
    lat: f64,
};

/// Parses a CSV file containing vertex data with the format: node_id,longitude,latitude
pub fn parseVertexCsv(allocator: std.mem.Allocator, file: std.fs.File) ![]Vertex {
    const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(content);

    var list = std.ArrayList(Vertex).empty;
    errdefer list.deinit(allocator);

    var lines = std.mem.splitScalar(u8, content, '\n');

    // Skip header line
    if (lines.next()) |header| {
        if (header.len == 0) return error.EmptyHeader;
    }

    while (lines.next()) |line| {
        if (line.len == 0) continue;

        var iter = std.mem.tokenizeScalar(u8, line, ',');
        const id_str = iter.next() orelse continue;
        const lon_str = iter.next() orelse continue;
        const lat_str = iter.next() orelse continue;

        // Check for additional unexpected fields
        if (iter.next() != null) return error.TooManyFields;

        const trimmed_id_str = std.mem.trim(u8, id_str, " \t\r\n");
        const trimmed_lon_str = std.mem.trim(u8, lon_str, " \t\r\n");
        const trimmed_lat_str = std.mem.trim(u8, lat_str, " \t\r\n");

        const vertex = Vertex{
            .id = try std.fmt.parseInt(u64, trimmed_id_str, 10),
            .lon = try std.fmt.parseFloat(f64, trimmed_lon_str),
            .lat = try std.fmt.parseFloat(f64, trimmed_lat_str),
        };

        try list.append(allocator, vertex);
    }

    return list.toOwnedSlice(allocator);
}

/// Finds the vertex closest to the given coordinates using Euclidean distance.
/// Returns the node ID of the closest vertex.
pub fn findClosestNode(
    allocator: std.mem.Allocator,
    target_lon: f64,
    target_lat: f64,
) !u64 {
    var file = try std.fs.cwd().openFile("data/vertex.csv", .{ .mode = .read_only });
    defer file.close();

    const vertices = try parseVertexCsv(allocator, file);
    defer allocator.free(vertices);

    if (vertices.len == 0) return error.NoVertices;

    var closest_id: u64 = vertices[0].id;
    var closest_dist_sq: f64 = std.math.inf(f64);

    for (vertices) |vertex| {
        const dx = vertex.lon - target_lon;
        const dy = vertex.lat - target_lat;
        const dist_sq = dx * dx + dy * dy;

        if (dist_sq < closest_dist_sq) {
            closest_dist_sq = dist_sq;
            closest_id = vertex.id;
        }
    }

    return closest_id;
}

/// Finds the closest vertex to the target coordinates and returns both the node ID and distance.
pub fn findClosestNodeWithDistance(
    allocator: std.mem.Allocator,
    target_lon: f64,
    target_lat: f64,
) !struct { id: u64, distance: f64 } {
    const id = try findClosestNode(allocator, target_lon, target_lat);

    var file = try std.fs.cwd().openFile("data/vertex.csv", .{ .mode = .read_only });
    defer file.close();

    const vertices = try parseVertexCsv(allocator, file);
    defer allocator.free(vertices);

    for (vertices) |vertex| {
        if (vertex.id == id) {
            const dx = vertex.lon - target_lon;
            const dy = vertex.lat - target_lat;
            const distance = std.math.sqrt(dx * dx + dy * dy);
            return .{ .id = id, .distance = distance };
        }
    }

    return error.VertexNotFound;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const target_lon: f64 = 126.60;
    const target_lat: f64 = 45.75;

    const result = try findClosestNodeWithDistance(allocator, target_lon, target_lat);
    std.debug.print("Closest node ID = {d}, distance = {d:.6} degrees\n", .{ result.id, result.distance });
}
