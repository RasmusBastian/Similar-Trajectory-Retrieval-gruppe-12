const std = @import("std");
const zap = @import("zap");
const endpoints = @import("endpoints.zig");
const errors = @import("errors.zig");

fn on_request_err(r: zap.Request, err: anyerror) !void {
    switch (err) {
        errors.ApiError.MethodNotAllowed => {
            r.setStatus(.method_not_allowed);
            try r.sendBody("Method not allowed \n");
        },
        errors.ApiError.InvalidJson => {
            r.setStatus(.bad_request);
            try r.sendBody("Invalid JSON \n");
        },
        errors.ApiError.MissingField => {
            r.setStatus(.bad_request);
            try r.sendBody("Missing field in JSON \n");
        },
        errors.ApiError.MissingBody => {
            r.setStatus(.bad_request);
            try r.sendBody("Missing request body \n");
        },
        else => {
            r.setStatus(.internal_server_error);
            try r.sendBody("Internal server error \n");
        },
    }
}

pub fn on_request(r: zap.Request) anyerror!void {
    const path = r.path orelse "/";

    // Add CORS headers to all responses
    r.setHeader("Access-Control-Allow-Origin", "*") catch {};
    r.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS") catch {};
    r.setHeader("Access-Control-Allow-Headers", "Content-Type") catch {};

    // Handle preflight OPTIONS request
    const method = r.method orelse "GET";
    if (std.mem.eql(u8, method, "OPTIONS")) {
        r.setStatus(.ok);
        try r.sendBody("");
        return;
    }


    const alloc = std.heap.page_allocator;

    // Handle the /docs path - serve API documentation
    if (std.mem.eql(u8, path, "/docs")) {
        const html_content = @embedFile("api-docs.html");
        r.setStatus(.ok);
        r.setHeader("Content-Type", "text/html; charset=utf-8") catch {};
        try r.sendBody(html_content);
        return;
    }

    // Handle the /api-spec.yml path - serve OpenAPI specification
    if (std.mem.eql(u8, path, "/api-spec.yml")) {
        const spec_content = @embedFile("api-spec.yml");
        r.setStatus(.ok);
        r.setHeader("Content-Type", "text/yaml; charset=utf-8") catch {};
        try r.sendBody(spec_content);
        return;
    }

    // Handle the /journey path
    if (std.mem.eql(u8, path, "/journey")) {
        const journey_response = endpoints.handleJourney(r, alloc) catch |err| {
            try on_request_err(r, err);
            return;
        };

        // Serialize response
        const json_string = std.json.Stringify.valueAlloc(alloc, journey_response, .{}) catch {
            r.setStatus(.internal_server_error);
            try r.sendBody("Failed to serialize response\n");
            return;
        };
        defer alloc.free(json_string);

        r.setStatus(.ok);
        r.setHeader("Content-Type", "application/json") catch {};
        try r.sendBody(json_string);

        alloc.free(journey_response.linestring.coordinates);

        return;
    }
}
