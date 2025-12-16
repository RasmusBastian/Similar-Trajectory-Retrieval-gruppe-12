const std = @import("std");
const lstm = @import("lstm.zig");
const LSTM = lstm.LSTM;
const TimeFeatures = lstm.TimeFeatures;

pub const Query = struct {
    edge_id: u64,
    time_offset: u64,
};

/// Convert a CSV row index to hour and minute
/// Each row represents a 5-minute time slot (288 rows per day)
/// Row 0 = 00:00, Row 1 = 00:05, Row 12 = 01:00, etc.
fn csvOffsetToTime(csv_offset: usize) struct { hour: u8, minute: u8 } {
    // Each row is 5 minutes, wrap around at 288 (full day)
    const slot = csv_offset % 288;
    const total_minutes = slot * 5;
    const hour: u8 = @intCast(total_minutes / 60);
    const minute: u8 = @intCast(total_minutes % 60);
    return .{ .hour = hour, .minute = minute };
}

/// Generate a lookup table based on a specific offset in the CSV data
/// offset: The line number in the CSV to use as the end point (e.g., 100 means we won't know anything beyond line 100)
/// The function will use sequence_length rows backward from the offset
/// For example, if offset=100 and sequence_length=3, it will use rows 98, 99, 100
pub fn generateLookupTable(
    allocator: std.mem.Allocator,
    model: *LSTM,
    csv_data: std.ArrayList(std.ArrayList(f32)),
    csv_offset: usize,
    future_lookout: usize,
) !std.AutoHashMap(Query, f32) {
    // Validate offset
    if (csv_offset >= csv_data.items.len) {
        return error.OffsetOutOfBounds;
    }
    if (csv_offset < model.sequence_length - 1) {
        return error.OffsetTooSmall;
    }

    // Prepare input data
    const input_data_size = model.batch_size * model.sequence_length * model.input_size;
    var input_data = try allocator.alloc(f32, input_data_size);
    defer allocator.free(input_data);

    // Fill input_data with sequence_length rows ending at offset
    // For example: if offset=100 and seq_length=3, use rows 98, 99, 100
    const start_row = csv_offset - (model.sequence_length - 1);
    var input_idx: usize = 0;
    for (start_row..csv_offset + 1) |row_idx| {
        const row = csv_data.items[row_idx];
        const copy_size = @min(row.items.len, model.input_size);
        @memcpy(input_data[input_idx .. input_idx + copy_size], row.items[0..copy_size]);
        input_idx += model.input_size;
    }

    // Create lookup table
    var lookup_table = std.AutoHashMap(Query, f32).init(allocator);
    errdefer lookup_table.deinit();

    // Allocate buffers for autoregressive prediction
    const current_window = try allocator.alloc(f32, input_data_size);
    defer allocator.free(current_window);

    const last_outputs = try allocator.alloc(f32, model.input_size);
    defer allocator.free(last_outputs);

    // Calculate time features for initial prediction (predicting csv_offset + 1)
    const initial_time = csvOffsetToTime(csv_offset + 1);
    const initial_time_features = TimeFeatures.fromTime(initial_time.hour, initial_time.minute);

    // Run initial inference and store at time_offset = 0
    const initial_output = try model.predict(allocator, input_data, initial_time_features);
    defer allocator.free(initial_output);

    // Store predictions for all edges (output dimensions)
    for (initial_output, 0..) |prediction, output_edge_id| {
        try lookup_table.put(Query{ .edge_id = output_edge_id, .time_offset = 0 }, prediction);
    }

    // Copy input data to current window for autoregressive prediction
    @memcpy(current_window, input_data);
    @memcpy(last_outputs, initial_output);

    // Generate future predictions
    for (1..future_lookout + 1) |time_offset| {
        // Shift window: remove first timestep, append last outputs
        shiftWindowStatic(current_window, last_outputs, model.sequence_length, model.input_size);

        // Calculate time features for this prediction step
        // time_offset=1 predicts csv_offset+2, time_offset=2 predicts csv_offset+3, etc.
        const future_time = csvOffsetToTime(csv_offset + 1 + time_offset);
        const future_time_features = TimeFeatures.fromTime(future_time.hour, future_time.minute);

        // Run inference and store
        const future_output = try model.predict(allocator, current_window, future_time_features);
        defer allocator.free(future_output);

        // Store predictions for all edges
        for (future_output, 0..) |prediction, output_edge_id| {
            try lookup_table.put(Query{ .edge_id = output_edge_id, .time_offset = time_offset }, prediction);
        }

        @memcpy(last_outputs, future_output);
    }

    return lookup_table;
}

/// Static version of shiftWindow for use in generateLookupTable
fn shiftWindowStatic(window: []f32, outputs: []f32, seq_len: usize, input_size: usize) void {
    // Shift all timesteps forward
    for (0..seq_len - 1) |i| {
        const src = (i + 1) * input_size;
        const dst = i * input_size;
        @memcpy(window[dst .. dst + input_size], window[src .. src + input_size]);
    }

    // Fill last timestep with all outputs
    const last_start = (seq_len - 1) * input_size;
    @memcpy(window[last_start .. last_start + input_size], outputs[0..input_size]);
}
