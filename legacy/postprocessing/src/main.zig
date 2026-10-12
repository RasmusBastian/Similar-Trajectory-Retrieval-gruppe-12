const std = @import("std");
const graph_module = @import("graph.zig");
const edge_csv_parser = @import("edge_csv_parser.zig");
const edge_connections_parser = @import("edge_connections_parser.zig");
const vertex_parser = @import("vertex_parser.zig");

// Configuration constants
const MAX_SPEED_KMH: f64 = 120.0;
const MIN_SPEED_KMH: f64 = 0.5;
const MAX_SPEED_MS: f64 = MAX_SPEED_KMH / 3.6;
const MIN_SPEED_MS: f64 = MIN_SPEED_KMH / 3.6;
const TEMPORAL_WINDOW: usize = 15; // slots before and after (31 total = 155 min)
const MISSING_VALUE: f32 = -1.0;
const SLOTS_PER_DAY: usize = 288;

const BadPoint = struct {
    edge_idx: usize,
    time_slot: usize,
};

/// Haversine formula - distance between two lat/lon points in meters
fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) f64 {
    const R = 6371000.0;
    const phi1 = lat1 * std.math.pi / 180.0;
    const phi2 = lat2 * std.math.pi / 180.0;
    const dphi = (lat2 - lat1) * std.math.pi / 180.0;
    const dlam = (lon2 - lon1) * std.math.pi / 180.0;

    const a = @sin(dphi / 2.0) * @sin(dphi / 2.0) +
        @cos(phi1) * @cos(phi2) * @sin(dlam / 2.0) * @sin(dlam / 2.0);
    const c = 2.0 * std.math.atan2(@sqrt(a), @sqrt(1.0 - a));

    return R * c;
}

/// Check if a data point is valid based on speed constraints
fn isValidSpeed(traversal_time: f32, edge_length_m: f64) bool {
    if (traversal_time <= 0 or traversal_time == MISSING_VALUE) return false;
    const speed_ms = edge_length_m / @as(f64, traversal_time);
    return speed_ms >= MIN_SPEED_MS and speed_ms <= MAX_SPEED_MS;
}

/// Calculate median of values
fn median(allocator: std.mem.Allocator, values: []const f32) !f32 {
    if (values.len == 0) return MISSING_VALUE;
    if (values.len == 1) return values[0];

    const sorted = try allocator.alloc(f32, values.len);
    defer allocator.free(sorted);
    @memcpy(sorted, values);
    std.mem.sort(f32, sorted, {}, std.sort.asc(f32));

    const mid = sorted.len / 2;
    if (sorted.len % 2 == 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
    return sorted[mid];
}

/// Context struct to hold all the data needed for aggressive fix
const FixContext = struct {
    all_edge_data: *std.ArrayList(std.ArrayList(f32)),
    edge_lengths: *std.AutoHashMap(u64, f64),
    edge_adjacency: *std.AutoHashMap(u64, std.ArrayList(u64)),
    edge_id_to_idx: *std.AutoHashMap(u64, usize),
    total_slots: usize,
    num_edges: usize,
    allocator: std.mem.Allocator,
};

/// Try to fix a bad point using temporal neighbors within window_size
fn tryTemporalFix(bad: BadPoint, window_size: usize, ctx: FixContext) ?f32 {
    const edge_id: u64 = @intCast(bad.edge_idx);
    const edge_length = ctx.edge_lengths.get(edge_id) orelse 100.0;

    var valid_values = std.ArrayList(f32).empty;
    defer valid_values.deinit(ctx.allocator);

    const start = if (bad.time_slot > window_size) bad.time_slot - window_size else 0;
    const end = @min(bad.time_slot + window_size + 1, ctx.total_slots);

    for (start..end) |t| {
        if (t == bad.time_slot) continue;

        const val = ctx.all_edge_data.items[t].items[bad.edge_idx];
        if (val != MISSING_VALUE and val > 0 and isValidSpeed(val, edge_length)) {
            valid_values.append(ctx.allocator, val) catch continue;
        }
    }

    if (valid_values.items.len >= 1) {
        return median(ctx.allocator, valid_values.items) catch return null;
    }
    return null;
}

/// Try to fix using spatial neighbors up to max_degree hops away
fn trySpatialFix(bad: BadPoint, max_degree: usize, ctx: FixContext) ?f32 {
    const edge_id: u64 = @intCast(bad.edge_idx);
    const edge_length = ctx.edge_lengths.get(edge_id) orelse 100.0;

    // Use BFS to collect edges at each degree
    var visited = std.AutoHashMap(u64, void).init(ctx.allocator);
    defer visited.deinit();
    visited.put(edge_id, {}) catch return null;

    var current_level = std.ArrayList(u64).empty;
    defer current_level.deinit(ctx.allocator);
    current_level.append(ctx.allocator, edge_id) catch return null;

    var weighted_sum: f64 = 0;
    var weight_total: f64 = 0;

    for (1..max_degree + 1) |degree| {
        var next_level = std.ArrayList(u64).empty;
        defer next_level.deinit(ctx.allocator);

        // Weight decreases with degree (1.0 for 1st degree, 0.5 for 2nd, 0.33 for 3rd, etc.)
        const degree_weight: f64 = 1.0 / @as(f64, @floatFromInt(degree));

        for (current_level.items) |curr_id| {
            const adjacent = ctx.edge_adjacency.get(curr_id) orelse continue;

            for (adjacent.items) |adj_id| {
                if (visited.contains(adj_id)) continue;
                visited.put(adj_id, {}) catch continue;

                const adj_idx = ctx.edge_id_to_idx.get(adj_id) orelse continue;
                if (adj_idx >= ctx.all_edge_data.items[bad.time_slot].items.len) continue;

                const adj_val = ctx.all_edge_data.items[bad.time_slot].items[adj_idx];
                if (adj_val == MISSING_VALUE or adj_val <= 0) {
                    next_level.append(ctx.allocator, adj_id) catch {};
                    continue;
                }

                const adj_length = ctx.edge_lengths.get(adj_id) orelse continue;
                if (!isValidSpeed(adj_val, adj_length)) {
                    next_level.append(ctx.allocator, adj_id) catch {};
                    continue;
                }

                // Weight by length similarity
                const length_ratio = if (edge_length > adj_length)
                    adj_length / edge_length
                else
                    edge_length / adj_length;

                // Scale traversal time proportionally to length
                const estimated_time = adj_val * @as(f32, @floatCast(edge_length / adj_length));

                // Only use if estimated time is valid for our edge
                if (isValidSpeed(estimated_time, edge_length)) {
                    weighted_sum += @as(f64, estimated_time) * length_ratio * degree_weight;
                    weight_total += length_ratio * degree_weight;
                }

                next_level.append(ctx.allocator, adj_id) catch {};
            }
        }

        // Move to next level
        current_level.clearRetainingCapacity();
        current_level.appendSlice(ctx.allocator, next_level.items) catch {};
    }

    if (weight_total > 0) {
        const result: f32 = @floatCast(weighted_sum / weight_total);
        if (isValidSpeed(result, edge_length)) {
            return result;
        }
    }
    return null;
}

/// Get average of all valid values for this edge across all time slots
fn getEdgeGlobalAverage(edge_idx: usize, ctx: FixContext) ?f32 {
    const edge_id: u64 = @intCast(edge_idx);
    const edge_length = ctx.edge_lengths.get(edge_id) orelse 100.0;

    var sum: f64 = 0;
    var count: usize = 0;

    for (ctx.all_edge_data.items) |row| {
        const val = row.items[edge_idx];
        if (val != MISSING_VALUE and val > 0 and isValidSpeed(val, edge_length)) {
            sum += @as(f64, val);
            count += 1;
        }
    }

    if (count > 0) {
        const avg: f32 = @floatCast(sum / @as(f64, @floatFromInt(count)));
        if (isValidSpeed(avg, edge_length)) {
            return avg;
        }
    }
    return null;
}

/// Get average from similar-length edges across the network (guaranteed to return a value)
fn getNetworkAverage(edge_idx: usize, ctx: FixContext) f32 {
    const edge_id: u64 = @intCast(edge_idx);
    const edge_length = ctx.edge_lengths.get(edge_id) orelse 100.0;

    // Find edges with similar length (±50% to be more permissive)
    const min_length = edge_length * 0.5;
    const max_length = edge_length * 1.5;

    var sum: f64 = 0;
    var count: usize = 0;

    var it = ctx.edge_lengths.iterator();
    while (it.next()) |entry| {
        const other_length = entry.value_ptr.*;
        if (other_length >= min_length and other_length <= max_length) {
            const other_idx = ctx.edge_id_to_idx.get(entry.key_ptr.*) orelse continue;

            for (ctx.all_edge_data.items) |row| {
                if (other_idx >= row.items.len) continue;
                const val = row.items[other_idx];
                if (val != MISSING_VALUE and val > 0 and isValidSpeed(val, other_length)) {
                    // Scale to our edge's length
                    const scaled = val * @as(f32, @floatCast(edge_length / other_length));
                    if (isValidSpeed(scaled, edge_length)) {
                        sum += @as(f64, scaled);
                        count += 1;
                    }
                }
            }
        }
    }

    if (count > 0) {
        return @floatCast(sum / @as(f64, @floatFromInt(count)));
    }

    // Ultimate fallback: use a reasonable default based on edge length
    // Assume average speed of 50 km/h = 13.89 m/s
    const default_speed: f64 = 50.0 / 3.6; // m/s
    return @floatCast(edge_length / default_speed);
}

/// Fix level tracking
const FixLevel = enum {
    temporal_15,
    temporal_30,
    temporal_60,
    temporal_144,
    temporal_288,
    temporal_all,
    spatial_3,
    spatial_4,
    spatial_5,
    spatial_6,
    spatial_10,
    global_avg,
    network_avg,
    default_speed,
};

/// Main aggressive fix function - keeps trying until it finds a solution
fn aggressiveFix(bad: BadPoint, ctx: FixContext) struct { value: f32, level: FixLevel } {
    const edge_id: u64 = @intCast(bad.edge_idx);
    const edge_length = ctx.edge_lengths.get(edge_id) orelse 100.0;

    // Level 1: temporal ±15 (already tried in main loop, but try again just in case)
    if (tryTemporalFix(bad, 15, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .temporal_15 };
    }

    // Level 2: temporal ±30
    if (tryTemporalFix(bad, 30, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .temporal_30 };
    }

    // Level 3: temporal ±60, spatial 3rd degree
    if (tryTemporalFix(bad, 60, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .temporal_60 };
    }
    if (trySpatialFix(bad, 3, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .spatial_3 };
    }

    // Level 4: temporal ±144 (half day), spatial 4th degree
    if (tryTemporalFix(bad, 144, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .temporal_144 };
    }
    if (trySpatialFix(bad, 4, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .spatial_4 };
    }

    // Level 5: temporal ±288 (full day), spatial 5th degree
    if (tryTemporalFix(bad, 288, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .temporal_288 };
    }
    if (trySpatialFix(bad, 5, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .spatial_5 };
    }

    // Level 6: temporal ALL, spatial 6th degree
    if (tryTemporalFix(bad, ctx.total_slots, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .temporal_all };
    }
    if (trySpatialFix(bad, 6, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .spatial_6 };
    }

    // Level 7: spatial 10th degree (essentially whole network)
    if (trySpatialFix(bad, 10, ctx)) |val| {
        if (isValidSpeed(val, edge_length)) return .{ .value = val, .level = .spatial_10 };
    }

    // Level 8: Global edge average
    if (getEdgeGlobalAverage(bad.edge_idx, ctx)) |val| {
        return .{ .value = val, .level = .global_avg };
    }

    // Level 9: Network average (always succeeds)
    const network_val = getNetworkAverage(bad.edge_idx, ctx);
    if (isValidSpeed(network_val, edge_length)) {
        return .{ .value = network_val, .level = .network_avg };
    }

    // Ultimate fallback: reasonable default speed
    const default_speed: f64 = 50.0 / 3.6; // 50 km/h in m/s
    const default_time: f32 = @floatCast(edge_length / default_speed);
    return .{ .value = default_time, .level = .default_speed };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize root progress node
    var root_progress = std.Progress.start(.{
        .root_name = "Postprocessing",
    });
    defer root_progress.end();

    std.debug.print("=== Edge Data Postprocessing ===\n\n", .{});

    // ===== Step 1: Load vertices and edge connections =====
    std.debug.print("Step 1: Loading graph data...\n", .{});

    const vertex_file = try std.fs.cwd().openFile("input/vertex.csv", .{ .mode = .read_only });
    defer vertex_file.close();
    const vertices = try vertex_parser.parseVertexCsv(allocator, vertex_file);
    defer allocator.free(vertices);
    std.debug.print("  Loaded {d} vertices\n", .{vertices.len});

    const edge_conn_file = try std.fs.cwd().openFile("input/edge_connections.csv", .{ .mode = .read_only });
    defer edge_conn_file.close();
    const edge_inits = try edge_connections_parser.parseEdgeConnectionsCsv(allocator, edge_conn_file);
    defer allocator.free(edge_inits);
    std.debug.print("  Loaded {d} edges\n", .{edge_inits.len});

    // Build the graph
    var graph = try graph_module.Graph.init(allocator, vertices, edge_inits);
    defer graph.deinit(allocator);
    std.debug.print("  Built graph with {d} nodes\n", .{graph.nodes.len});

    // ===== Step 2: Load edge data for all days =====
    std.debug.print("\nStep 2: Loading edge traversal data...\n", .{});

    const day_files = [_][]const u8{
        "input/edge_data_day3.csv",
        "input/edge_data_day4.csv",
        "input/edge_data_day5.csv",
        "input/edge_data_day6.csv",
        "input/edge_data_day7.csv",
    };

    var all_edge_data = std.ArrayList(std.ArrayList(f32)).empty;
    defer {
        for (all_edge_data.items) |*row| row.deinit(allocator);
        all_edge_data.deinit(allocator);
    }

    for (day_files, 0..) |day_file, day_idx| {
        const file = std.fs.cwd().openFile(day_file, .{ .mode = .read_only }) catch |err| {
            std.debug.print("  Warning: Could not open {s}: {}\n", .{ day_file, err });
            continue;
        };
        defer file.close();

        var day_data = try edge_csv_parser.parseEdgeCsv(allocator, file);
        defer {
            for (day_data.items) |*row| row.deinit(allocator);
            day_data.deinit(allocator);
        }

        for (day_data.items) |row| {
            var new_row = std.ArrayList(f32).empty;
            try new_row.appendSlice(allocator, row.items);
            try all_edge_data.append(allocator, new_row);
        }
        std.debug.print("  Day {d}: {d} time slots\n", .{ day_idx + 3, day_data.items.len });
    }

    const total_slots = all_edge_data.items.len;
    const num_edges = if (total_slots > 0) all_edge_data.items[0].items.len else 0;
    std.debug.print("  Total: {d} time slots x {d} edges\n", .{ total_slots, num_edges });

    // ===== Step 3: Calculate edge lengths and build adjacency =====
    std.debug.print("\nStep 3: Computing edge lengths and adjacency...\n", .{});

    // Map edge_id -> length_m
    var edge_lengths = std.AutoHashMap(u64, f64).init(allocator);
    defer edge_lengths.deinit();

    // Map edge_id -> edge_idx (position in CSV columns)
    var edge_id_to_idx = std.AutoHashMap(u64, usize).init(allocator);
    defer edge_id_to_idx.deinit();

    // Map vertex_id -> list of edge_ids touching it
    var vertex_to_edges = std.AutoHashMap(u64, std.ArrayList(u64)).init(allocator);
    defer {
        var it = vertex_to_edges.valueIterator();
        while (it.next()) |list| list.deinit(allocator);
        vertex_to_edges.deinit();
    }

    // Iterate through graph to collect edge info
    for (graph.nodes, 0..) |node, node_idx| {
        for (graph.adj[node_idx]) |edge| {
            // Calculate length using haversine
            if (graph.id_to_index.get(edge.to)) |to_idx| {
                const to_node = graph.nodes[to_idx];
                const length = haversine(node.lat, node.lon, to_node.lat, to_node.lon);
                try edge_lengths.put(edge.edge_id, length);

                // Map edge_id to its CSV column index (edge_id is 0-indexed)
                try edge_id_to_idx.put(edge.edge_id, @intCast(edge.edge_id));

                // Track which edges touch each vertex
                const start_entry = try vertex_to_edges.getOrPut(node.id);
                if (!start_entry.found_existing) {
                    start_entry.value_ptr.* = std.ArrayList(u64).empty;
                }
                try start_entry.value_ptr.append(allocator, edge.edge_id);

                const end_entry = try vertex_to_edges.getOrPut(edge.to);
                if (!end_entry.found_existing) {
                    end_entry.value_ptr.* = std.ArrayList(u64).empty;
                }
                try end_entry.value_ptr.append(allocator, edge.edge_id);
            }
        }
    }
    std.debug.print("  Computed {d} edge lengths\n", .{edge_lengths.count()});

    // Build edge_id -> adjacent_edge_ids map
    var edge_adjacency = std.AutoHashMap(u64, std.ArrayList(u64)).init(allocator);
    defer {
        var it = edge_adjacency.valueIterator();
        while (it.next()) |list| list.deinit(allocator);
        edge_adjacency.deinit();
    }

    for (graph.nodes, 0..) |node, node_idx| {
        for (graph.adj[node_idx]) |edge| {
            var adjacent = std.ArrayList(u64).empty;

            // Get edges from start vertex (node.id)
            if (vertex_to_edges.get(node.id)) |edges_at_start| {
                for (edges_at_start.items) |adj_id| {
                    if (adj_id != edge.edge_id) {
                        try adjacent.append(allocator, adj_id);
                    }
                }
            }

            // Get edges from end vertex (edge.to)
            if (vertex_to_edges.get(edge.to)) |edges_at_end| {
                for (edges_at_end.items) |adj_id| {
                    if (adj_id != edge.edge_id) {
                        // Avoid duplicates
                        var found = false;
                        for (adjacent.items) |existing| {
                            if (existing == adj_id) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) try adjacent.append(allocator, adj_id);
                    }
                }
            }

            try edge_adjacency.put(edge.edge_id, adjacent);
        }
    }
    std.debug.print("  Built adjacency for {d} edges\n", .{edge_adjacency.count()});

    // ===== Step 4: Identify bad data points =====
    std.debug.print("\nStep 4: Detecting bad data points...\n", .{});

    var bad_points = std.ArrayList(BadPoint).empty;
    defer bad_points.deinit(allocator);

    var bad_set = std.AutoHashMap(u64, void).init(allocator);
    defer bad_set.deinit();

    var missing_count: usize = 0;
    var too_fast: usize = 0;
    var too_slow: usize = 0;

    var detect_progress = root_progress.start("Detecting bad points", total_slots);
    for (all_edge_data.items, 0..) |row, time_slot| {
        detect_progress.completeOne();
        for (row.items, 0..) |traversal_time, edge_idx| {
            const edge_id: u64 = @intCast(edge_idx);
            const length = edge_lengths.get(edge_id) orelse 100.0;

            const is_bad = if (traversal_time == MISSING_VALUE or traversal_time <= 0) blk: {
                missing_count += 1;
                break :blk true;
            } else if (!isValidSpeed(traversal_time, length)) blk: {
                const speed_kmh = (length / @as(f64, traversal_time)) * 3.6;
                if (speed_kmh > MAX_SPEED_KMH) {
                    too_fast += 1;
                } else {
                    too_slow += 1;
                }
                break :blk true;
            } else false;

            if (is_bad) {
                try bad_points.append(allocator, .{ .edge_idx = edge_idx, .time_slot = time_slot });
                const key = @as(u64, @intCast(time_slot)) * num_edges + edge_idx;
                try bad_set.put(key, {});
            }
        }
    }

    detect_progress.end();
    std.debug.print("  Found {d} bad points (speed violations):\n", .{bad_points.items.len});
    std.debug.print("    Missing/invalid: {d}\n", .{missing_count});
    std.debug.print("    Too fast (>{d} km/h): {d}\n", .{ MAX_SPEED_KMH, too_fast });
    std.debug.print("    Too slow (<{d} km/h): {d}\n", .{ MIN_SPEED_KMH, too_slow });

    // ===== Step 4b: Detect temporal outliers (spikes) =====
    std.debug.print("\nStep 4b: Detecting temporal outliers...\n", .{});

    var outlier_count: usize = 0;
    const OUTLIER_WINDOW: usize = 6; // Look at ±6 slots (30 min each side)
    const IQR_MULTIPLIER: f32 = 2.0; // Points outside Q1 - 2*IQR or Q3 + 2*IQR are outliers

    var outlier_progress = root_progress.start("Detecting outliers", num_edges);
    for (0..num_edges) |edge_idx| {
        outlier_progress.completeOne();

        // Collect all valid values for this edge to calculate IQR
        var all_valid = std.ArrayList(f32).empty;
        defer all_valid.deinit(allocator);

        for (all_edge_data.items) |row| {
            const val = row.items[edge_idx];
            if (val != MISSING_VALUE and val > 0) {
                all_valid.append(allocator, val) catch continue;
            }
        }

        // Need at least 10 values to calculate meaningful IQR
        if (all_valid.items.len < 10) continue;

        // Sort to find quartiles
        std.mem.sort(f32, all_valid.items, {}, std.sort.asc(f32));

        const q1_idx = all_valid.items.len / 4;
        const q3_idx = (all_valid.items.len * 3) / 4;
        const q1 = all_valid.items[q1_idx];
        const q3 = all_valid.items[q3_idx];
        const iqr = q3 - q1;

        // Skip if IQR is too small (data is very uniform)
        if (iqr < 1.0) continue;

        const lower_bound = q1 - IQR_MULTIPLIER * iqr;
        const upper_bound = q3 + IQR_MULTIPLIER * iqr;

        // Now check each time slot for this edge
        for (all_edge_data.items, 0..) |row, time_slot| {
            const val = row.items[edge_idx];
            if (val == MISSING_VALUE or val <= 0) continue;

            // Skip if already marked as bad
            const key = @as(u64, @intCast(time_slot)) * num_edges + edge_idx;
            if (bad_set.contains(key)) continue;

            // Check if it's an outlier based on global IQR
            if (val < lower_bound or val > upper_bound) {
                // Double-check with local window to avoid marking legitimate traffic changes
                // (e.g., rush hour vs night)
                var local_values = std.ArrayList(f32).empty;
                defer local_values.deinit(allocator);

                const start = if (time_slot > OUTLIER_WINDOW) time_slot - OUTLIER_WINDOW else 0;
                const end = @min(time_slot + OUTLIER_WINDOW + 1, total_slots);

                for (start..end) |t| {
                    if (t == time_slot) continue;
                    const local_val = all_edge_data.items[t].items[edge_idx];
                    if (local_val != MISSING_VALUE and local_val > 0) {
                        local_values.append(allocator, local_val) catch continue;
                    }
                }

                if (local_values.items.len < 3) continue;

                // Calculate local median
                std.mem.sort(f32, local_values.items, {}, std.sort.asc(f32));
                const local_median = local_values.items[local_values.items.len / 2];

                // If value differs from local median by more than 50%, it's a spike
                const ratio = if (val > local_median) val / local_median else local_median / val;
                if (ratio > 1.5) {
                    try bad_points.append(allocator, .{ .edge_idx = edge_idx, .time_slot = time_slot });
                    try bad_set.put(key, {});
                    outlier_count += 1;
                }
            }
        }
    }
    outlier_progress.end();

    std.debug.print("  Found {d} temporal outliers (spikes)\n", .{outlier_count});
    std.debug.print("  Total bad points: {d}\n", .{bad_points.items.len});

    // ===== Step 5: Aggressive Iterative Interpolation =====
    std.debug.print("\nStep 5: Applying aggressive interpolation (100%% fix rate)...\n", .{});

    // Create fix context
    const ctx = FixContext{
        .all_edge_data = &all_edge_data,
        .edge_lengths = &edge_lengths,
        .edge_adjacency = &edge_adjacency,
        .edge_id_to_idx = &edge_id_to_idx,
        .total_slots = total_slots,
        .num_edges = num_edges,
        .allocator = allocator,
    };

    // Counters per fix level
    var fix_counts = [_]usize{0} ** 14;
    const level_names = [_][]const u8{
        "temporal_15",
        "temporal_30",
        "temporal_60",
        "temporal_144",
        "temporal_288",
        "temporal_all",
        "spatial_3",
        "spatial_4",
        "spatial_5",
        "spatial_6",
        "spatial_10",
        "global_avg",
        "network_avg",
        "default_speed",
    };

    var fix_progress = root_progress.start("Aggressive fix", bad_points.items.len);
    for (bad_points.items) |bad| {
        fix_progress.completeOne();
        const result = aggressiveFix(bad, ctx);
        all_edge_data.items[bad.time_slot].items[bad.edge_idx] = result.value;
        fix_counts[@intFromEnum(result.level)] += 1;
    }
    fix_progress.end();

    // Report fix statistics
    std.debug.print("  Fix breakdown by level:\n", .{});
    var total_fixed: usize = 0;
    for (fix_counts, 0..) |count, i| {
        if (count > 0) {
            std.debug.print("    {s}: {d}\n", .{ level_names[i], count });
        }
        total_fixed += count;
    }
    std.debug.print("  Total fixed: {d} (100%%)\n", .{total_fixed});

    // ===== Step 6: Write cleaned data =====
    std.debug.print("\nStep 6: Writing cleaned data...\n", .{});

    std.fs.cwd().makeDir("output") catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    for (0..5) |day_idx| {
        const start_slot = day_idx * SLOTS_PER_DAY;
        const end_slot = @min(start_slot + SLOTS_PER_DAY, total_slots);
        if (start_slot >= total_slots) break;

        var filename_buf: [64]u8 = undefined;
        const filename = try std.fmt.bufPrint(&filename_buf, "output/edge_data_day{d}_cleaned.csv", .{day_idx + 3});

        const out_file = try std.fs.cwd().createFile(filename, .{});
        defer out_file.close();

        // Build content in memory using ArrayList
        var content = std.ArrayList(u8).empty;
        defer content.deinit(allocator);

        // Header
        try content.appendSlice(allocator, "time_slot");
        for (0..num_edges) |i| {
            var col_buf: [64]u8 = undefined;
            const col_str = try std.fmt.bufPrint(&col_buf, ",edge{d}_traversal_time_sec", .{i});
            try content.appendSlice(allocator, col_str);
        }
        try content.appendSlice(allocator, "\n");

        // Data rows
        for (start_slot..end_slot) |slot| {
            const slot_in_day = slot - start_slot;
            const hours = slot_in_day * 5 / 60;
            const minutes = (slot_in_day * 5) % 60;

            var time_buf: [16]u8 = undefined;
            const time_str = try std.fmt.bufPrint(&time_buf, "{d:0>2}:{d:0>2}", .{ hours, minutes });
            try content.appendSlice(allocator, time_str);

            for (all_edge_data.items[slot].items) |value| {
                var val_buf: [32]u8 = undefined;
                const val_str = try std.fmt.bufPrint(&val_buf, ",{d}", .{value});
                try content.appendSlice(allocator, val_str);
            }
            try content.appendSlice(allocator, "\n");
        }

        try out_file.writeAll(content.items);
        std.debug.print("  Written: {s}\n", .{filename});
    }

    // Summary
    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("Total bad points: {d}\n", .{bad_points.items.len});
    std.debug.print("Total fixed: {d} (100%%)\n", .{total_fixed});
    std.debug.print("Fix breakdown:\n", .{});
    for (fix_counts, 0..) |count, i| {
        if (count > 0) {
            std.debug.print("  {s}: {d}\n", .{ level_names[i], count });
        }
    }
}

