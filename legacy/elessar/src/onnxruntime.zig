
const std = @import("std");
const Allocator = std.mem.Allocator;

/// API docs: https://onnxruntime.ai/docs/api/c/struct_ort_api.html
pub const raw = @import("onnxruntime_raw");

pub const ONNXTensorElementDataType = enum(u32) {
    undefined = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    bool = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    string = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, // maps to c++ type std::string
    f16 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    f32 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // maps to c type float
    f64 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, // maps to c type double
    bf16 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, // Non-IEEE floating-point format based on IEEE754 single-precision
    u8 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, // maps to c type uint8_t
    u16 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, // maps to c type uint16_t
    u32 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, // maps to c type uint32_t
    u64 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, // maps to c type uint64_t
    i8 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, // maps to c type int8_t
    i16 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, // maps to c type int16_t
    i32 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, // maps to c type int32_t
    i64 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, // maps to c type int64_t
    c64 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64, // complex with float32 real and imaginary components
    c128 = raw.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128, // complex with float64 real and imaginary components
};

pub const OrtAllocatorType = enum(i32) {
    invalid = raw.OrtInvalidAllocator,
    device = raw.OrtDeviceAllocator,
    arena = raw.OrtArenaAllocator,
};

pub const OrtMemType = enum(i32) {
    /// The default allocator for execution provider
    default = raw.OrtMemTypeDefault,
    /// Any CPU memory used by non-CPU execution provider
    cpu_input = raw.OrtMemTypeCPUInput,
    /// CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
    cpu_output = raw.OrtMemTypeCPUOutput,
};

pub const OrtLoggingLevel = enum(u32) {
    verbose = raw.ORT_LOGGING_LEVEL_VERBOSE,
    info = raw.ORT_LOGGING_LEVEL_INFO,
    warning = raw.ORT_LOGGING_LEVEL_WARNING,
    @"error" = raw.ORT_LOGGING_LEVEL_ERROR,
    fatal = raw.ORT_LOGGING_LEVEL_FATAL,
};

pub fn checkError(
    ort_api: *const raw.OrtApi,
    onnx_status: ?*raw.OrtStatus,
) !void {
    if (onnx_status == null) return;
    defer ort_api.ReleaseStatus.?(onnx_status);

    const msg = ort_api.GetErrorMessage.?(onnx_status);
    std.debug.print("ONNX error: {s}\n", .{msg});

    return error.OnnxError;
}

pub const OrtEnv = struct {
    raw: *raw.OrtEnv,

    pub fn init(ort_api: *const raw.OrtApi, logging_level: OrtLoggingLevel, log_id: *[:0]const u8) !OrtEnv {
        var ort_env: ?*raw.OrtEnv = null;
        const status = ort_api.CreateEnv.?(
            @intFromEnum(logging_level),
            log_id.ptr,
            &ort_env,
        );

        try checkError(ort_api, status);
        return OrtEnv{ .raw = ort_env.? };
    }

    pub fn deinit(self: *OrtEnv, ort_api: *const raw.OrtApi) void {
        ort_api.ReleaseEnv.?(self.raw);
    }
};

pub const GraphOptimizationLevel = enum(c_int) {
    disable_all = raw.ORT_DISABLE_ALL,
    enable_basic = raw.ORT_ENABLE_BASIC,
    enable_extended = raw.ORT_ENABLE_EXTENDED,
    enable_all = raw.ORT_ENABLE_ALL,
};

pub const ExecutionMode = enum(c_int) {
    sequential = raw.ORT_SEQUENTIAL,
    parallel = raw.ORT_PARALLEL,
};

pub const CUDAProviderOptions = struct {
    device_id: c_int = 0,
    gpu_mem_limit: ?usize = null, // null = use all available
    arena_extend_strategy: ArenaExtendStrategy = .next_power_of_two,
    cudnn_conv_algo_search: CudnnConvAlgoSearch = .default,
    do_copy_in_default_stream: bool = true,
    has_user_compute_stream: bool = false,
    user_compute_stream: ?*anyopaque = null,
    tunable_op_enable: bool = false,
    tunable_op_tuning_enable: bool = false,
    tunable_op_max_tuning_duration_ms: i32 = 0,

    pub const ArenaExtendStrategy = enum(u1) {
        next_power_of_two = 0,
        same_as_requested = 1,
    };

    pub const CudnnConvAlgoSearch = enum(u2) {
        exhaustive = 0,
        heuristic = 1,
        default = 2,
    };

    /// Apply these options to a session
    pub fn apply(self: CUDAProviderOptions, allocator: Allocator, ort_api: *const raw.OrtApi, session_opts: *raw.OrtSessionOptions) !void {
        // Create ORT CUDA options
        var cuda_options: ?*raw.OrtCUDAProviderOptionsV2 = null;
        {
            const status = ort_api.CreateCUDAProviderOptions.?(&cuda_options);
            try checkError(ort_api, status);
        }
        defer ort_api.ReleaseCUDAProviderOptions.?(cuda_options);

        // Convert Zig struct to string key-value pairs
        var keys = try std.ArrayList([*:0]const u8).initCapacity(allocator, 100);
        defer keys.deinit(allocator);
        var values = try std.ArrayList([*:0]const u8).initCapacity(allocator, 100);
        defer values.deinit(allocator);

        // Allocate buffers for string conversions
        var buffers = try std.ArrayList([]u8).initCapacity(allocator, 100);
        defer {
            for (buffers.items) |buf| {
                allocator.free(buf);
            }
            buffers.deinit(allocator);
        }

        // Helper to add option with allocated string
        const addOption = struct {
            fn add(
                k: *std.ArrayList([*:0]const u8),
                v: *std.ArrayList([*:0]const u8),
                bufs: *std.ArrayList([]u8),
                alloc: Allocator,
                key: [:0]const u8,
                value: []const u8,
            ) !void {
                const val_z = try alloc.dupeZ(u8, value);
                try bufs.append(alloc, val_z);
                try k.append(alloc, key.ptr);
                try v.append(alloc, val_z.ptr);
            }
        }.add;

        // device_id
        {
            var buf: [32]u8 = undefined;
            const device_id_str = try std.fmt.bufPrint(&buf, "{d}", .{self.device_id});
            try addOption(&keys, &values, &buffers, allocator, "device_id", device_id_str);
        }

        // gpu_mem_limit (if specified)
        if (self.gpu_mem_limit) |limit| {
            var buf: [32]u8 = undefined;
            const limit_str = try std.fmt.bufPrint(&buf, "{d}", .{limit});
            try addOption(&keys, &values, &buffers, allocator, "gpu_mem_limit", limit_str);
        }

        // arena_extend_strategy - DISABLED: Not supported in ONNX Runtime v1.22.2+
        // {
        //     var buf: [8]u8 = undefined;
        //     const arena_str = try std.fmt.bufPrint(&buf, "{d}", .{@intFromEnum(self.arena_extend_strategy)});
        //     try addOption(&keys, &values, &buffers, allocator, "arena_extend_strategy", arena_str);
        // }

        // cudnn_conv_algo_search - DISABLED: Not supported in ONNX Runtime v1.22.2+
        // {
        //     var buf: [8]u8 = undefined;
        //     const cudnn_str = try std.fmt.bufPrint(&buf, "{d}", .{@intFromEnum(self.cudnn_conv_algo_search)});
        //     try addOption(&keys, &values, &buffers, allocator, "cudnn_conv_algo_search", cudnn_str);
        // }

        // booleans
        try addOption(&keys, &values, &buffers, allocator, "do_copy_in_default_stream", if (self.do_copy_in_default_stream) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "tunable_op_enable", if (self.tunable_op_enable) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "tunable_op_tuning_enable", if (self.tunable_op_tuning_enable) "1" else "0");

        if (self.tunable_op_max_tuning_duration_ms > 0) {
            var buf: [32]u8 = undefined;
            const duration_str = try std.fmt.bufPrint(&buf, "{d}", .{self.tunable_op_max_tuning_duration_ms});
            try addOption(&keys, &values, &buffers, allocator, "tunable_op_max_tuning_duration_ms", duration_str);
        }

        // Update options
        {
            const status = ort_api.UpdateCUDAProviderOptions.?(
                cuda_options,
                keys.items.ptr,
                values.items.ptr,
                keys.items.len,
            );
            try checkError(ort_api, status);
        }

        // Apply to session
        {
            const status = ort_api.SessionOptionsAppendExecutionProvider_CUDA_V2.?(
                session_opts,
                cuda_options,
            );
            try checkError(ort_api, status);
        }
    }
};

pub const TensorRTProviderOptions = struct {
    device_id: c_int = 0,
    max_workspace_size: usize = 1 << 30, // 1GB default
    fp16_enable: bool = false,
    int8_enable: bool = false,
    max_partition_iterations: i32 = 1000,
    min_subgraph_size: i32 = 1,
    dla_enable: bool = false,
    dla_core: i32 = 0,
    engine_cache_enable: bool = false,
    engine_cache_path: ?[:0]const u8 = null,
    int8_calibration_table_name: ?[:0]const u8 = null,
    int8_use_native_calibration_table: bool = false,
    dump_subgraphs: bool = false,
    engine_decryption_enable: bool = false,
    engine_decryption_lib_path: ?[:0]const u8 = null,
    force_sequential_engine_build: bool = false,

    /// Apply these options to a session
    pub fn apply(self: TensorRTProviderOptions, allocator: Allocator, ort_api: *const raw.OrtApi, session_opts: *raw.OrtSessionOptions) !void {
        // Create ORT TensorRT options
        var trt_options: ?*raw.OrtTensorRTProviderOptionsV2 = null;
        {
            const status = ort_api.CreateTensorRTProviderOptions.?(&trt_options);
            try checkError(ort_api, status);
        }
        defer ort_api.ReleaseTensorRTProviderOptions.?(trt_options);

        // Convert Zig struct to string key-value pairs
        var keys = try std.ArrayList([*:0]const u8).initCapacity(allocator, 100);
        defer keys.deinit(allocator);
        var values = try std.ArrayList([*:0]const u8).initCapacity(allocator, 100);
        defer values.deinit(allocator);

        // Allocate buffers for string conversions
        var buffers = try std.ArrayList([]u8).initCapacity(allocator, 100);
        defer {
            for (buffers.items) |buf| {
                allocator.free(buf);
            }
            buffers.deinit(allocator);
        }

        // Helper to add option with allocated string
        const addOption = struct {
            fn add(
                k: *std.ArrayList([*:0]const u8),
                v: *std.ArrayList([*:0]const u8),
                bufs: *std.ArrayList([]u8),
                alloc: Allocator,
                key: [:0]const u8,
                value: []const u8,
            ) !void {
                const val_z = try alloc.dupeZ(u8, value);
                try bufs.append(alloc, val_z);
                try k.append(alloc, key.ptr);
                try v.append(alloc, val_z.ptr);
            }
        }.add;

        // device_id
        {
            var buf: [32]u8 = undefined;
            const device_id_str = try std.fmt.bufPrint(&buf, "{d}", .{self.device_id});
            try addOption(&keys, &values, &buffers, allocator, "device_id", device_id_str);
        }

        // max_workspace_size
        {
            var buf: [32]u8 = undefined;
            const workspace_str = try std.fmt.bufPrint(&buf, "{d}", .{self.max_workspace_size});
            try addOption(&keys, &values, &buffers, allocator, "trt_max_workspace_size", workspace_str);
        }

        // max_partition_iterations
        {
            var buf: [32]u8 = undefined;
            const iter_str = try std.fmt.bufPrint(&buf, "{d}", .{self.max_partition_iterations});
            try addOption(&keys, &values, &buffers, allocator, "trt_max_partition_iterations", iter_str);
        }

        // min_subgraph_size
        {
            var buf: [32]u8 = undefined;
            const size_str = try std.fmt.bufPrint(&buf, "{d}", .{self.min_subgraph_size});
            try addOption(&keys, &values, &buffers, allocator, "trt_min_subgraph_size", size_str);
        }

        // booleans
        try addOption(&keys, &values, &buffers, allocator, "trt_fp16_enable", if (self.fp16_enable) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "trt_int8_enable", if (self.int8_enable) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "trt_dla_enable", if (self.dla_enable) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "trt_engine_cache_enable", if (self.engine_cache_enable) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "trt_dump_subgraphs", if (self.dump_subgraphs) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "trt_engine_decryption_enable", if (self.engine_decryption_enable) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "trt_force_sequential_engine_build", if (self.force_sequential_engine_build) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "trt_int8_use_native_calibration_table", if (self.int8_use_native_calibration_table) "1" else "0");

        // dla_core
        if (self.dla_enable) {
            var buf: [32]u8 = undefined;
            const dla_str = try std.fmt.bufPrint(&buf, "{d}", .{self.dla_core});
            try addOption(&keys, &values, &buffers, allocator, "trt_dla_core", dla_str);
        }

        // optional paths
        if (self.engine_cache_path) |path| {
            try addOption(&keys, &values, &buffers, allocator, "trt_engine_cache_path", path);
        }
        if (self.int8_calibration_table_name) |name| {
            try addOption(&keys, &values, &buffers, allocator, "trt_int8_calibration_table_name", name);
        }
        if (self.engine_decryption_lib_path) |path| {
            try addOption(&keys, &values, &buffers, allocator, "trt_engine_decryption_lib_path", path);
        }

        // Update options
        {
            const status = ort_api.UpdateTensorRTProviderOptions.?(
                trt_options,
                keys.items.ptr,
                values.items.ptr,
                keys.items.len,
            );
            try checkError(ort_api, status);
        }

        // Apply to session
        {
            const status = ort_api.SessionOptionsAppendExecutionProvider_TensorRT_V2.?(
                session_opts,
                trt_options,
            );
            try checkError(ort_api, status);
        }
    }
};

pub const ROCMProviderOptions = struct {
    device_id: c_int = 0,
    gpu_mem_limit: ?usize = null,
    arena_extend_strategy: CUDAProviderOptions.ArenaExtendStrategy = .next_power_of_two,
    miopen_conv_exhaustive_search: bool = false,
    do_copy_in_default_stream: bool = true,
    tunable_op_enable: bool = false,
    tunable_op_tuning_enable: bool = false,
    tunable_op_max_tuning_duration_ms: i32 = 0,
    enable_hip_graph: bool = false,

    /// Apply these options to a session
    pub fn apply(self: ROCMProviderOptions, allocator: Allocator, ort_api: *const raw.OrtApi, session_opts: *raw.OrtSessionOptions) !void {
        // Create ORT ROCM options
        var rocm_options: ?*raw.OrtROCMProviderOptions = null;
        {
            const status = ort_api.CreateROCMProviderOptions.?(&rocm_options);
            try checkError(ort_api, status);
        }
        defer ort_api.ReleaseROCMProviderOptions.?(rocm_options);

        // Convert Zig struct to string key-value pairs
        var keys = std.ArrayList([*:0]const u8).init(allocator);
        defer keys.deinit();
        var values = std.ArrayList([*:0]const u8).init(allocator);
        defer values.deinit();

        // Allocate buffers for string conversions
        var buffers = std.ArrayList([]u8).init(allocator);
        defer {
            for (buffers.items) |buf| {
                allocator.free(buf);
            }
            buffers.deinit();
        }

        // Helper to add option
        const addOption = struct {
            fn add(
                k: *std.ArrayList([*:0]const u8),
                v: *std.ArrayList([*:0]const u8),
                bufs: *std.ArrayList([]u8),
                alloc: Allocator,
                key: [:0]const u8,
                value: []const u8,
            ) !void {
                const val_z = try alloc.dupeZ(u8, value);
                try bufs.append(val_z);
                try k.append(key.ptr);
                try v.append(val_z.ptr);
            }
        }.add;

        // device_id
        {
            var buf: [32]u8 = undefined;
            const device_id_str = try std.fmt.bufPrint(&buf, "{d}", .{self.device_id});
            try addOption(&keys, &values, &buffers, allocator, "device_id", device_id_str);
        }

        // gpu_mem_limit (if specified)
        if (self.gpu_mem_limit) |limit| {
            var buf: [32]u8 = undefined;
            const limit_str = try std.fmt.bufPrint(&buf, "{d}", .{limit});
            try addOption(&keys, &values, &buffers, allocator, "gpu_mem_limit", limit_str);
        }

        // arena_extend_strategy - DISABLED: Not supported in ONNX Runtime v1.22.2+
        // {
        //     var buf: [8]u8 = undefined;
        //     const arena_str = try std.fmt.bufPrint(&buf, "{d}", .{@intFromEnum(self.arena_extend_strategy)});
        //     try addOption(&keys, &values, &buffers, allocator, "arena_extend_strategy", arena_str);
        // }

        // booleans
        try addOption(&keys, &values, &buffers, allocator, "miopen_conv_exhaustive_search", if (self.miopen_conv_exhaustive_search) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "do_copy_in_default_stream", if (self.do_copy_in_default_stream) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "tunable_op_enable", if (self.tunable_op_enable) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "tunable_op_tuning_enable", if (self.tunable_op_tuning_enable) "1" else "0");
        try addOption(&keys, &values, &buffers, allocator, "enable_hip_graph", if (self.enable_hip_graph) "1" else "0");

        if (self.tunable_op_max_tuning_duration_ms > 0) {
            var buf: [32]u8 = undefined;
            const duration_str = try std.fmt.bufPrint(&buf, "{d}", .{self.tunable_op_max_tuning_duration_ms});
            try addOption(&keys, &values, &buffers, allocator, "tunable_op_max_tuning_duration_ms", duration_str);
        }

        // Update options
        {
            const status = ort_api.UpdateROCMProviderOptions.?(
                rocm_options,
                keys.items.ptr,
                values.items.ptr,
                keys.items.len,
            );
            try checkError(ort_api, status);
        }

        // Apply to session
        {
            const status = ort_api.SessionOptionsAppendExecutionProvider_ROCM.?(
                session_opts,
                rocm_options,
            );
            try checkError(ort_api, status);
        }
    }
};

/// Fluent builder for creating OrtSessionOptions with type-safe configuration
///
/// Example usage:
/// ```zig
/// const ort_api = getOrtApi();
/// var builder = try SessionOptionsBuilder().init(allocator, ort_api);
/// defer builder.deinit();
///
/// const session_opts = try builder
///     .withCUDA(.{ .device_id = 0, .tunable_op_enable = true })
///     .withIntraOpNumThreads(4)
///     .withGraphOptimizationLevel(.enable_all)
///     .build();
/// ```
pub const OrtSessionOptions = struct {
    const Self = @This();

    allocator: Allocator,
    ort_api: *const raw.OrtApi,
    session_opts: *raw.OrtSessionOptions,

    pub fn init(allocator: Allocator, ort_api: *const raw.OrtApi) !Self {
        var opts: ?*raw.OrtSessionOptions = null;
        const status = ort_api.CreateSessionOptions.?(&opts);
        try checkError(ort_api, status);

        return Self{
            .allocator = allocator,
            .ort_api = ort_api,
            .session_opts = opts.?,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ort_api.ReleaseSessionOptions.?(self.session_opts);
    }

    pub fn withCUDA(self: *Self, cuda_opts: CUDAProviderOptions) !*Self {
        try cuda_opts.apply(self.allocator, self.ort_api, self.session_opts);
        return self;
    }

    pub fn withTensorRT(self: *Self, trt_opts: TensorRTProviderOptions) !*Self {
        try trt_opts.apply(self.allocator, self.ort_api, self.session_opts);
        return self;
    }

    pub fn withROCM(self: *Self, rocm_opts: ROCMProviderOptions) !*Self {
        try rocm_opts.apply(self.allocator, self.ort_api, self.session_opts);
        return self;
    }

    pub fn withIntraOpNumThreads(self: *Self, num_threads: i32) !*Self {
        const status = self.ort_api.SetIntraOpNumThreads.?(self.session_opts, num_threads);
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withInterOpNumThreads(self: *Self, num_threads: i32) !*Self {
        const status = self.ort_api.SetInterOpNumThreads.?(self.session_opts, num_threads);
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withGraphOptimizationLevel(self: *Self, level: GraphOptimizationLevel) !*Self {
        const status = self.ort_api.SetSessionGraphOptimizationLevel.?(
            self.session_opts,
            @as(c_uint, @intCast(@intFromEnum(level))),
        );
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withExecutionMode(self: *Self, mode: ExecutionMode) !*Self {
        const status = self.ort_api.SetSessionExecutionMode.?(
            self.session_opts,
            @intFromEnum(mode),
        );
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withLogId(self: *Self, log_id: [:0]const u8) !*Self {
        const status = self.ort_api.SetSessionLogId.?(self.session_opts, log_id.ptr);
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withLogVerbosityLevel(self: *Self, level: i32) !*Self {
        const status = self.ort_api.SetSessionLogVerbosityLevel.?(self.session_opts, level);
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withOptimizedModelFilePath(self: *Self, path: [:0]const raw.ORTCHAR_T) !*Self {
        const status = self.ort_api.SetOptimizedModelFilePath.?(self.session_opts, path.ptr);
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withProfiling(self: *Self, profile_file_prefix: [:0]const raw.ORTCHAR_T) !*Self {
        const status = self.ort_api.EnableProfiling.?(self.session_opts, profile_file_prefix.ptr);
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withMemPattern(self: *Self, enable: bool) !*Self {
        const status = if (enable)
            self.ort_api.EnableMemPattern.?(self.session_opts)
        else
            self.ort_api.DisableMemPattern.?(self.session_opts);
        try checkError(self.ort_api, status);
        return self;
    }

    pub fn withCpuMemArena(self: *Self, enable: bool) !*Self {
        const status = if (enable)
            self.ort_api.EnableCpuMemArena.?(self.session_opts)
        else
            self.ort_api.DisableCpuMemArena.?(self.session_opts);
        try checkError(self.ort_api, status);
        return self;
    }
};

pub const OrtSession = struct {
    raw: *raw.OrtSession,
    ort_api: *const raw.OrtApi,

    pub fn init(
        ort_api: *const raw.OrtApi,
        env: *OrtEnv,
        model_path: [:0]const u8,
        session_opts: *OrtSessionOptions,
    ) !OrtSession {
        var ort_sess: ?*raw.OrtSession = null;
        const status = ort_api.CreateSession.?(
            env.raw,
            model_path.ptr,
            session_opts.session_opts,
            &ort_sess,
        );

        try checkError(ort_api, status);
        return OrtSession{
            .raw = ort_sess.?,
            .ort_api = ort_api,
        };
    }

    pub fn deinit(self: *OrtSession) void {
        self.ort_api.ReleaseSession.?(self.raw);
    }

    /// Run inference on the session
    pub fn run(
        self: *OrtSession,
        run_opts: ?*raw.OrtRunOptions,
        input_names: [][*:0]const u8,
        inputs: []*raw.OrtValue,
        output_names: [][*:0]const u8,
        outputs: []?*raw.OrtValue,
    ) !void {
        const status = self.ort_api.Run.?(
            self.raw,
            run_opts,
            input_names.ptr,
            inputs.ptr,
            inputs.len,
            output_names.ptr,
            output_names.len,
            outputs.ptr,
        );
        try checkError(self.ort_api, status);
    }
};

/// Get the ONNX Runtime API
pub fn getOrtAPI() *const raw.OrtApi {
    return raw.OrtGetApiBase().*.GetApi.?(raw.ORT_API_VERSION);
}

pub const OrtRunOptions = struct {
    raw: *raw.OrtRunOptions,
    ort_api: *const raw.OrtApi,

    pub fn init(ort_api: *const raw.OrtApi) !OrtRunOptions {
        var run_opts: ?*raw.OrtRunOptions = null;
        const status = ort_api.CreateRunOptions.?(&run_opts);
        try checkError(ort_api, status);

        return OrtRunOptions{
            .raw = run_opts.?,
            .ort_api = ort_api,
        };
    }

    pub fn deinit(self: *OrtRunOptions) void {
        self.ort_api.ReleaseRunOptions.?(self.raw);
    }

    pub fn setRunLogVerbosityLevel(self: *OrtRunOptions, level: i32) !void {
        const status = self.ort_api.RunOptionsSetRunLogVerbosityLevel.?(self.raw, level);
        try checkError(self.ort_api, status);
    }

    pub fn setRunTag(self: *OrtRunOptions, tag: [:0]const u8) !void {
        const status = self.ort_api.RunOptionsSetRunTag.?(self.raw, tag.ptr);
        try checkError(self.ort_api, status);
    }

    pub fn setTerminate(self: *OrtRunOptions) !void {
        const status = self.ort_api.RunOptionsSetTerminate.?(self.raw);
        try checkError(self.ort_api, status);
    }

    pub fn unsetTerminate(self: *OrtRunOptions) !void {
        const status = self.ort_api.RunOptionsUnsetTerminate.?(self.raw);
        try checkError(self.ort_api, status);
    }
};

pub const OrtMemoryInfo = struct {
    raw: *raw.OrtMemoryInfo,
    ort_api: *const raw.OrtApi,

    pub fn init(
        ort_api: *const raw.OrtApi,
        name: [:0]const u8,
        allocator_type: OrtAllocatorType,
        device_id: i32,
        mem_type: OrtMemType,
    ) !OrtMemoryInfo {
        var mem_info: ?*raw.OrtMemoryInfo = null;
        const status = ort_api.CreateMemoryInfo.?(
            name.ptr,
            @intFromEnum(allocator_type),
            device_id,
            @intFromEnum(mem_type),
            &mem_info,
        );
        try checkError(ort_api, status);

        return OrtMemoryInfo{
            .raw = mem_info.?,
            .ort_api = ort_api,
        };
    }

    pub fn deinit(self: *OrtMemoryInfo) void {
        self.ort_api.ReleaseMemoryInfo.?(self.raw);
    }
};

pub const OrtValue = struct {
    raw: *raw.OrtValue,
    ort_api: *const raw.OrtApi,

    /// Create a tensor backed by user-supplied buffer (zero-copy)
    pub fn createTensorWithData(
        ort_api: *const raw.OrtApi,
        mem_info: *OrtMemoryInfo,
        comptime T: type,
        data: []T,
        shape: []const i64,
        tensor_type: ONNXTensorElementDataType,
    ) !OrtValue {
        var value: ?*raw.OrtValue = null;
        const status = ort_api.CreateTensorWithDataAsOrtValue.?(
            mem_info.raw,
            data.ptr,
            data.len * @sizeOf(T),
            shape.ptr,
            shape.len,
            @intFromEnum(tensor_type),
            &value,
        );
        try checkError(ort_api, status);

        return OrtValue{
            .raw = value.?,
            .ort_api = ort_api,
        };
    }

    pub fn deinit(self: *OrtValue) void {
        self.ort_api.ReleaseValue.?(self.raw);
    }

    /// Get mutable pointer to tensor data
    pub fn getTensorMutableData(self: *OrtValue, comptime T: type) ![]T {
        var data_ptr: ?*anyopaque = undefined;
        const status = self.ort_api.GetTensorMutableData.?(self.raw, &data_ptr);
        try checkError(self.ort_api, status);

        if (data_ptr == null) return error.NoTensorData;

        // Get tensor type info to determine size
        var type_info: ?*raw.OrtTypeInfo = undefined;
        {
            const get_type_status = self.ort_api.GetTypeInfo.?(self.raw, &type_info);
            try checkError(self.ort_api, get_type_status);
        }
        defer self.ort_api.ReleaseTypeInfo.?(type_info);

        var tensor_info: ?*const raw.OrtTensorTypeAndShapeInfo = undefined;
        {
            const get_tensor_status = self.ort_api.CastTypeInfoToTensorInfo.?(type_info, &tensor_info);
            try checkError(self.ort_api, get_tensor_status);
        }

        var element_count: usize = undefined;
        {
            const get_count_status = self.ort_api.GetTensorShapeElementCount.?(tensor_info, &element_count);
            try checkError(self.ort_api, get_count_status);
        }

        const typed_ptr: [*]T = @ptrCast(@alignCast(data_ptr.?));
        return typed_ptr[0..element_count];
    }

    /// Check if value is a tensor
    pub fn isTensor(self: *OrtValue) !bool {
        var is_tensor: c_int = 0;
        const status = self.ort_api.IsTensor.?(self.raw, &is_tensor);
        try checkError(self.ort_api, status);
        return is_tensor != 0;
    }
};

