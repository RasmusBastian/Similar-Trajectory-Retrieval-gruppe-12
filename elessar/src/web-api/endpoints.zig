const std = @import("std");

const astar = @import("../astar.zig");
const zap = @import("zap");

const main = @import("../main.zig");
const errors = @import("errors.zig");
const generateLookupTable = @import("../lookup_table.zig").generateLookupTable;
const types = @import("types.zig");
const graphtypes = @import("../graph_types.zig");

fn handleJourneyPost(r: zap.Request, requestAllocator: std.mem.Allocator) !types.JourneyResponse {
    const body = r.body orelse return errors.ApiError.MissingBody;

    // Create allocator for temporary data (JSON parsing, lookup table, etc.)
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Validate JSON request structure
    const parsedBody = std.json.parseFromSliceLeaky(types.JourneyInput, allocator, body, .{
        .ignore_unknown_fields = false,
    }) catch {
        return errors.ApiError.InvalidJson;
    };

    // Get global data (loaded once in main.zig)
    const edge_data = main.global_edge_data orelse return error.DataNotInitialized;
    const graph = main.global_graph orelse return error.DataNotInitialized;
    const model = main.global_model orelse return error.ModelNotInitialized;

    // Find closest nodes to start/end coordinates using graph's preloaded nodes
    const startNodeId = findClosestNodeInMemory(graph.nodes, parsedBody.start_coordinate.lon, parsedBody.start_coordinate.lat);
    const endNodeId = findClosestNodeInMemory(graph.nodes, parsedBody.end_coordinate.lon, parsedBody.end_coordinate.lat);

    std.debug.print("Start: ({d}, {d}) -> node {}\n", .{ parsedBody.start_coordinate.lat, parsedBody.start_coordinate.lon, startNodeId });
    std.debug.print("End: ({d}, {d}) -> node {}\n", .{ parsedBody.end_coordinate.lat, parsedBody.end_coordinate.lon, endNodeId });

    // Convert time to CSV offset (each row = 5 minutes, 12 rows per hour)
    // Must be at least sequence_length - 1 (29) for the model
    const min_offset: usize = 29;
    const time_offset = @as(usize, parsedBody.time.hour) * 12 + @as(usize, parsedBody.time.minute) / 5;
    const csv_offset = @max(time_offset, min_offset);

    // Lock the model mutex to make sure we have thread_safe access
    main.global_model_mutex.lock();
    defer main.global_model_mutex.unlock();

    // Generate Lookup table with csv data and model
    var lookup_table = try generateLookupTable(allocator, model, edge_data.*, csv_offset, 40);
    defer lookup_table.deinit();

    // Debug: print some sample lookup table values
    std.debug.print("Lookup table size: {}\n", .{lookup_table.count()});
    var sample_count: usize = 0;
    var it = lookup_table.iterator();
    while (it.next()) |entry| {
        if (sample_count < 10) {
            std.debug.print("  edge_id={}, time_offset={} -> {d:.2}s\n", .{ entry.key_ptr.edge_id, entry.key_ptr.time_offset, entry.value_ptr.* });
            sample_count += 1;
        }
    }

    // Run astar
    const timestep_seconds = 300.0; // Each timestep in the CSV represents 5 minutes

    const journey = try astar.aStar(allocator, graph.*, startNodeId, endNodeId, &lookup_table, timestep_seconds);

    // Build response coordinates using graph's id_to_index for O(1) lookup
    var arr = std.ArrayList([2]f32){};
    for (journey.stops) |stop| {
        const idx = graph.id_to_index.get(stop) orelse return error.NodeNotFound;
        const node = graph.nodes[idx];
        const coords: [2]f32 = .{ @floatCast(node.lat), @floatCast(node.lon) };
        try arr.append(requestAllocator, coords);
    }

    return types.JourneyResponse{
        .linestring = .{ .type = "LineString", .coordinates = arr.items },
        .traversalTime = @floatFromInt(journey.estimated_traversal_time_seconds),
        .length = journey.length_m / 1000,
    };
}

/// Find closest node to coordinates using preloaded vertices (no file I/O)
fn findClosestNodeInMemory(vertices: []const graphtypes.Node, target_lon: f64, target_lat: f64) u64 {
    var closest_id: u64 = vertices[0].id;
    var closest_dist_sq: f64 = std.math.floatMax(f64);

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

pub fn handleJourney(r: zap.Request, alloc: std.mem.Allocator) !types.JourneyResponse {
    const method = r.method orelse "GET";

    if (!std.mem.eql(u8, method, "POST")) {
        return errors.ApiError.MethodNotAllowed;
    }

    std.debug.print("POST request for {?s}\n", .{r.path});
    return try handleJourneyPost(r, alloc);
}
