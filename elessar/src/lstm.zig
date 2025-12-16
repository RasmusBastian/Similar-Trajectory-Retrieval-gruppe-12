const std = @import("std");
const onnx = @import("onnxruntime");

/// LSTM inference wrapper for ONNX Runtime with time feature support
///
/// The model includes PCA transforms embedded, so inputs/outputs are in original feature space.
///
/// Example usage:
/// ```zig
/// const allocator = std.heap.page_allocator;
///
/// // Initialize ONNX Runtime
/// const ort_api = onnx.getOrtAPI();
/// var log_id: [:0]const u8 = "lstm";
/// var ort_env = try onnx.OrtEnv.init(ort_api, .warning, &log_id);
/// defer ort_env.deinit(ort_api);
///
/// // Create session options
/// var session_opts = try onnx.OrtSessionOptions.init(allocator, ort_api);
/// defer session_opts.deinit();
///
/// // Initialize LSTM model (with PCA embedded - input/output in original space)
/// var lstm = try LSTM.init(allocator, &ort_env, &session_opts, .{
///     .model_path = "data/best_model.onnx",
///     .batch_size = 1,
///     .sequence_length = 24,
///     .input_size = 21312, // Number of edges (original feature space)
/// });
/// defer lstm.deinit();
///
/// // Prepare input data [batch_size, sequence_length, input_size]
/// var input_data = try allocator.alloc(f32, 1 * 24 * 21312);
/// defer allocator.free(input_data);
/// // ... fill input_data with edge travel times ...
///
/// // Run inference with time features (hour=14, minute=30)
/// const output = try lstm.predictWithTime(allocator, input_data, 14, 30);
/// defer allocator.free(output);
///
/// // Or use TimeFeatures directly:
/// const time_features = TimeFeatures.fromTime(14, 30);
/// const output2 = try lstm.predict(allocator, input_data, time_features);
/// defer allocator.free(output2);
/// ```
/// Time features for LSTM model input
/// Contains cyclical encodings of hour and minute
pub const TimeFeatures = struct {
    hour_sin: f32,
    hour_cos: f32,
    minute_sin: f32,
    minute_cos: f32,

    /// Generate time features from hour (0-23) and minute (0-59)
    /// Matches Python's extract_time_features() implementation
    pub fn fromTime(hour: u8, minute: u8) TimeFeatures {
        const hour_f: f32 = @floatFromInt(hour);
        const minute_f: f32 = @floatFromInt(minute);

        // Cyclical encoding for hour (24-hour cycle)
        const hour_rad = 2.0 * std.math.pi * hour_f / 24.0;

        // Cyclical encoding for minute of day (1440-minute cycle)
        // This matches Python: minutes_of_day = hour * 60 + minute
        const minutes_of_day = hour_f * 60.0 + minute_f;
        const minute_rad = 2.0 * std.math.pi * minutes_of_day / 1440.0;

        return TimeFeatures{
            .hour_sin = @sin(hour_rad),
            .hour_cos = @cos(hour_rad),
            .minute_sin = @sin(minute_rad),
            .minute_cos = @cos(minute_rad),
        };
    }

    /// Convert to array for tensor creation
    pub fn toArray(self: TimeFeatures) [4]f32 {
        return [_]f32{ self.hour_sin, self.hour_cos, self.minute_sin, self.minute_cos };
    }
};

pub const LSTM = struct {
    allocator: std.mem.Allocator,
    ort_api: *const onnx.raw.OrtApi,
    session: onnx.OrtSession,
    mem_info: onnx.OrtMemoryInfo,
    run_opts: onnx.OrtRunOptions,

    // Input/output names
    sequence_input_name: [*:0]const u8,
    time_features_input_name: [*:0]const u8,
    output_name: [*:0]const u8,

    // Expected dimensions
    batch_size: usize,
    sequence_length: usize,
    input_size: usize,
    time_feature_size: usize,

    pub const InitOptions = struct {
        model_path: [:0]const u8,
        batch_size: usize,
        sequence_length: usize,
        input_size: usize,
        time_feature_size: usize = 4, // Default: hour_sin, hour_cos, minute_sin, minute_cos
    };

    pub fn init(
        allocator: std.mem.Allocator,
        ort_env: *onnx.OrtEnv,
        session_opts: *onnx.OrtSessionOptions,
        options: InitOptions,
    ) !LSTM {
        const ort_api = onnx.getOrtAPI();

        // Create session
        var session = try onnx.OrtSession.init(
            ort_api,
            ort_env,
            options.model_path,
            session_opts,
        );
        errdefer session.deinit();

        // Create memory info for CPU
        var mem_info = try onnx.OrtMemoryInfo.init(ort_api, "Cpu", .arena, 0, .default);
        errdefer mem_info.deinit();

        // Create run options
        var run_opts = try onnx.OrtRunOptions.init(ort_api);
        errdefer run_opts.deinit();

        return LSTM{
            .allocator = allocator,
            .ort_api = ort_api,
            .session = session,
            .mem_info = mem_info,
            .run_opts = run_opts,
            .sequence_input_name = "sequence",
            .time_features_input_name = "time_features",
            .output_name = "prediction",
            .batch_size = options.batch_size,
            .sequence_length = options.sequence_length,
            .input_size = options.input_size,
            .time_feature_size = options.time_feature_size,
        };
    }

    pub fn deinit(self: *LSTM) void {
        self.run_opts.deinit();
        self.mem_info.deinit();
        self.session.deinit();
    }

    /// Run inference on input sequence data with time features
    /// Input data should be in shape [batch_size, sequence_length, input_size]
    /// Time features contain cyclical hour/minute encodings for the prediction target
    /// Returns prediction output
    pub fn predict(
        self: *LSTM,
        allocator: std.mem.Allocator,
        input_data: []const f32,
        time_features: TimeFeatures,
    ) ![]f32 {
        // Validate input size
        const expected_input_size = self.batch_size * self.sequence_length * self.input_size;
        if (input_data.len != expected_input_size) {
            return error.InputSizeMismatch;
        }

        // Create sequence input tensor [batch_size, sequence_length, input_size]
        const sequence_shape = [_]i64{
            @intCast(self.batch_size),
            @intCast(self.sequence_length),
            @intCast(self.input_size),
        };

        const input_data_copy = try allocator.alloc(f32, input_data.len);
        defer allocator.free(input_data_copy);
        @memcpy(input_data_copy, input_data);

        var sequence_value = try onnx.OrtValue.createTensorWithData(
            self.ort_api,
            &self.mem_info,
            f32,
            input_data_copy,
            &sequence_shape,
            .f32,
        );
        defer sequence_value.deinit();

        // Create time features tensor [batch_size, time_feature_size]
        const time_features_shape = [_]i64{
            @intCast(self.batch_size),
            @intCast(self.time_feature_size),
        };

        var time_features_array = time_features.toArray();
        var time_features_value = try onnx.OrtValue.createTensorWithData(
            self.ort_api,
            &self.mem_info,
            f32,
            &time_features_array,
            &time_features_shape,
            .f32,
        );
        defer time_features_value.deinit();

        var input_names = [_][*:0]const u8{ self.sequence_input_name, self.time_features_input_name };
        var output_names = [_][*:0]const u8{self.output_name};
        var inputs_array = [_]*onnx.raw.OrtValue{ sequence_value.raw, time_features_value.raw };
        var outputs_array = [_]?*onnx.raw.OrtValue{null};

        // Run inference
        try self.session.run(
            self.run_opts.raw,
            &input_names,
            &inputs_array,
            &output_names,
            &outputs_array,
        );

        // Get output tensor
        if (outputs_array[0] == null) {
            return error.NoOutputTensor;
        }

        var output_value = onnx.OrtValue{
            .raw = outputs_array[0].?,
            .ort_api = self.ort_api,
        };
        defer output_value.deinit();

        // Get output data
        const output_tensor_data = try output_value.getTensorMutableData(f32);

        // Copy output data
        const output_data = try allocator.alloc(f32, output_tensor_data.len);
        @memcpy(output_data, output_tensor_data);

        return output_data;
    }

    /// Convenience method to predict with hour and minute values
    pub fn predictWithTime(
        self: *LSTM,
        allocator: std.mem.Allocator,
        input_data: []const f32,
        hour: u8,
        minute: u8,
    ) ![]f32 {
        const time_features = TimeFeatures.fromTime(hour, minute);
        return self.predict(allocator, input_data, time_features);
    }
};

// Test to load an ONNX model and run inference
// Note: This test requires the model file to exist at the specified path
test "LSTM inference test" {
    const allocator = std.testing.allocator;

    // Model with PCA embedded - input/output in original feature space
    const model_path = "data/best_model.onnx";
    const batch_size: usize = 1;
    const sequence_length: usize = 24;
    const input_size: usize = 21312; // Number of edges (original feature space)

    // Initialize ONNX Runtime environment
    const ort_api = onnx.getOrtAPI();
    var log_id: [:0]const u8 = "lstm_test";
    var ort_env = try onnx.OrtEnv.init(ort_api, .warning, &log_id);
    defer ort_env.deinit(ort_api);

    // Create session options
    var session_opts = try onnx.OrtSessionOptions.init(allocator, ort_api);
    defer session_opts.deinit();

    _ = try session_opts.withGraphOptimizationLevel(.enable_all);
    _ = try session_opts.withIntraOpNumThreads(4);

    // Initialize LSTM
    var model = try LSTM.init(allocator, &ort_env, &session_opts, .{
        .model_path = model_path,
        .batch_size = batch_size,
        .sequence_length = sequence_length,
        .input_size = input_size,
    });
    defer model.deinit();

    // Create dummy input data [batch_size, sequence_length, input_size]
    const input_data_size = batch_size * sequence_length * input_size;
    const input_data = try allocator.alloc(f32, input_data_size);
    defer allocator.free(input_data);

    // Fill with sample data (simulating edge travel times)
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (input_data) |*val| {
        // Random travel times between 10-300 seconds
        val.* = random.float(f32) * 290.0 + 10.0;
    }

    // Run inference with time features (e.g., 14:30)
    const output = try model.predictWithTime(allocator, input_data, 14, 30);
    defer allocator.free(output);

    // Verify output size matches input_size (number of edges)
    try std.testing.expectEqual(input_size, output.len);
}

test "TimeFeatures generation" {
    // Test midnight (00:00) - minutes_of_day = 0
    const midnight = TimeFeatures.fromTime(0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), midnight.hour_sin, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), midnight.hour_cos, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), midnight.minute_sin, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), midnight.minute_cos, 0.0001);

    // Test 6:00 AM (hour_sin should be 1, hour_cos should be 0)
    // minutes_of_day = 360, 360/1440 = 0.25, sin(0.5*pi) = 1, cos(0.5*pi) = 0
    const six_am = TimeFeatures.fromTime(6, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), six_am.hour_sin, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), six_am.hour_cos, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), six_am.minute_sin, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), six_am.minute_cos, 0.0001);

    // Test 12:00 (noon - hour_sin should be 0, hour_cos should be -1)
    // minutes_of_day = 720, 720/1440 = 0.5, sin(pi) = 0, cos(pi) = -1
    const noon = TimeFeatures.fromTime(12, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), noon.hour_sin, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), noon.hour_cos, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), noon.minute_sin, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), noon.minute_cos, 0.0001);

    // Test 18:00 (6 PM) - minutes_of_day = 1080, 1080/1440 = 0.75
    // sin(1.5*pi) = -1, cos(1.5*pi) = 0
    const six_pm = TimeFeatures.fromTime(18, 0);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), six_pm.hour_sin, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), six_pm.hour_cos, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), six_pm.minute_sin, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), six_pm.minute_cos, 0.0001);
}
