const std = @import("std");

pub fn parseEdgeCsv(allocator: std.mem.Allocator, file: std.fs.File) !std.ArrayList(std.ArrayList(f32)) {
    const content = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(content);

    var outer_list = std.ArrayList(std.ArrayList(f32)).empty;
    errdefer {
        for (outer_list.items) |*list| {
            list.deinit(allocator);
        }
        outer_list.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, content, '\n');

    // Skip the first line with names
    _ = lines.next();

    while (lines.next()) |line| {
        if (line.len == 0) continue;

        var list = std.ArrayList(f32).empty;
        errdefer list.deinit(allocator);

        var iter = std.mem.tokenizeScalar(u8, line, ',');

        // Skip the first column (time_slot)
        _ = iter.next();

        while (iter.next()) |field| {
            const num = try std.fmt.parseFloat(f32, field);
            try list.append(allocator, num);
        }

        try outer_list.append(allocator, list);
    }

    return outer_list;
}
