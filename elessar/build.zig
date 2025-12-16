const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const onnxruntime = b.dependency("onnxruntime", .{});
    const onnxruntime_raw = b.addTranslateC(.{
        .root_source_file = onnxruntime.path("include/onnxruntime/core/session/onnxruntime_c_api.h"),
        .optimize = optimize,
        .target = target,
        .link_libc = true,
    });
    const onnxruntime_raw_module = onnxruntime_raw.createModule();

    // Create ONNX Runtime module wrapper
    const onnxruntime_mod = b.addModule("onnxruntime", .{
        .root_source_file = b.path("src/onnxruntime.zig"),
        .target = target,
        .imports = &.{
            .{ .name = "onnxruntime_raw", .module = onnxruntime_raw_module },
        },
    });

    const mod = b.addModule("elessar", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // Web API only executable (no onnxruntime dependency)
    const exe_web_api = b.addExecutable(.{
        .name = "elessar-web",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "elessar", .module = mod },
            },
        }),
    });

    // Full executable with onnxruntime support
    const exe = b.addExecutable(.{
        .name = "elessar",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "elessar", .module = mod },
                .{ .name = "onnxruntime", .module = onnxruntime_mod },
            },
        }),
    });

    exe.root_module.addLibraryPath(.{ .cwd_relative = "./onnxruntime-linux-x64/lib" });
    exe.root_module.addIncludePath(.{ .cwd_relative = "./onnxruntime-linux-x64/include" });
    exe.linkSystemLibrary("onnxruntime");

    const zap = b.dependency("zap", .{
        .target = target,
        .optimize = optimize,
        .openssl = false, // set to true to enable TLS support
    });

    exe.root_module.addImport("zap", zap.module("zap"));
    exe_web_api.root_module.addImport("zap", zap.module("zap"));

    // Install full version to default install step
    b.installArtifact(exe);

    // Install web API separately
    const install_web = b.addInstallArtifact(exe_web_api, .{});

    // Run step for full version (with onnxruntime)
    const run_step = b.step("run", "Run the full app (requires onnxruntime)");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Run step for web API only (no onnxruntime)
    const run_web_step = b.step("run-web", "Run the web API only (no ML model)");
    const run_web_cmd = b.addRunArtifact(exe_web_api);
    run_web_step.dependOn(&run_web_cmd.step);
    run_web_cmd.step.dependOn(&install_web.step);
    if (b.args) |args| {
        run_web_cmd.addArgs(args);
    }

    // Creates an executable that will run `test` blocks from the provided module.
    // Here `mod` needs to define a target, which is why earlier we made sure to
    // set the releative field.
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const lstm_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lstm.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "onnxruntime", .module = onnxruntime_mod },
            },
        }),
    });

    lstm_tests.linkSystemLibrary("onnxruntime");
    const run_lstm_tests = b.addRunArtifact(lstm_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_lstm_tests.step);
}
