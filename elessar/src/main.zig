const std = @import("std");

const zap = @import("zap");
const handler = @import("web-api/handler.zig");

const astar = @import("astar.zig");

const generateLookupTable = @import("lookup_table.zig").generateLookupTable;
const Query = @import("lookup_table.zig").Query;

const parseEdgeCsv = @import("edge_csv_parser.zig").parseEdgeCsv;
const parseEdgeConnectionsCsv = @import("edge_connections_parser.zig").parseEdgeConnectionsCsv;
const EdgeConnection = @import("edge_connections_parser.zig").EdgeConnection;

const parseVertexCsv = @import("vertex_parser.zig").parseVertexCsv;
const Vertex = @import("vertex_parser.zig").Vertex;

const onnx = @import("onnxruntime");
const LSTM = @import("lstm.zig").LSTM;
const graphtypes = @import("graph_types.zig");

// Global model state - initialized once, shared across requests
pub var global_model: ?*LSTM = null;
pub var global_model_mutex: std.Thread.Mutex = .{};
var global_ort_env: ?onnx.OrtEnv = null;
var global_session_opts: ?onnx.OrtSessionOptions = null;

// Global CSV data - loaded once at startup
pub var global_edge_data: ?*std.ArrayList(std.ArrayList(f32)) = null;
pub var global_edge_connections: ?[]graphtypes.EdgeInit = null;
pub var global_vertices: ?[]graphtypes.Node = null;

// Global graph - built once at startup
pub var global_graph: ?*astar.Graph = null;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load CSV data (once)
    std.debug.print("Loading CSV data...\n", .{});

    const edge_data_file = try std.fs.cwd().openFile("data/edge_data_day7.csv", .{});
    defer edge_data_file.close();
    var edge_data = try parseEdgeCsv(allocator, edge_data_file);
    global_edge_data = &edge_data;
    defer {
        for (edge_data.items) |*row| {
            row.deinit(allocator);
        }
        edge_data.deinit(allocator);
        global_edge_data = null;
    }

    const edge_connections_file = try std.fs.cwd().openFile("data/edge_connections.csv", .{});
    defer edge_connections_file.close();
    const edge_connections = try parseEdgeConnectionsCsv(allocator, edge_connections_file);
    global_edge_connections = edge_connections;
    defer {
        allocator.free(edge_connections);
        global_edge_connections = null;
    }

    const vertices_file = try std.fs.cwd().openFile("data/vertex.csv", .{});
    defer vertices_file.close();
    const vertices = try parseVertexCsv(allocator, vertices_file);
    global_vertices = vertices;
    defer {
        allocator.free(vertices);
        global_vertices = null;
    }

    std.debug.print("Loaded {} edge data rows, {} edge connections, {} vertices\n", .{
        edge_data.items.len,
        edge_connections.len,
        vertices.len,
    });

    // Build graph (once)
    std.debug.print("Building graph...\n", .{});
    var graph = try astar.Graph.init(allocator, vertices, edge_connections);
    global_graph = &graph;
    defer {
        graph.deinit(allocator);
        global_graph = null;
    }
    std.debug.print("Graph built with {} nodes\n", .{graph.nodes.len});

    // Initialize ONNX Runtime environment (once)
    const ort_api = onnx.getOrtAPI();
    var log_id: [:0]const u8 = "lstm";
    global_ort_env = try onnx.OrtEnv.init(ort_api, .warning, &log_id);
    defer global_ort_env.?.deinit(ort_api);

    global_session_opts = try onnx.OrtSessionOptions.init(allocator, ort_api);
    defer global_session_opts.?.deinit();

    // Initialize LSTM model (once)
    var model = try LSTM.init(allocator, &global_ort_env.?, &global_session_opts.?, .{
        .model_path = "data/best_model.onnx",
        .batch_size = 1,
        .sequence_length = 21,
        .input_size = 22748,
    });
    defer model.deinit();
    global_model = &model;
    defer {
        global_model = null;
    }

    std.debug.print("LSTM model initialized\n", .{});

    var listener = zap.HttpListener.init(.{
        .port = 3030,
        .on_request = handler.on_request,
        .log = true,
        .max_clients = 100000,
    });
    try listener.listen();

    std.debug.print("Listening on 0.0.0.0:3030\n", .{});

    // start worker threads
    zap.start(.{
        .threads = 2,
        .workers = 1, // 1 worker enables sharing state between threads
    });

    //// START EXAMPLE API CALL WHERE WE GET CSV OFFSET, START_NODE, END_NODE

    // _ = some_astar_idunno(start_node, end_node, lookup_table, &graph);
    // TODO:  Return the route to the api consumer

    //// END EXAMPLE API CALL WHERE WE GET CSV OFFSET, START_NODE, END_NODE

    // Here we try and get the value for edge 5 at time_offset 0. This will then be at index 30 (02:30) in the data.
    // If we tried time_offset at 10, we would get the value for index 40 (03:20)
    // const edge_val = try lookup_tabl.get(.{ .edge_id = 5, .time_offset = 0 });
    // std.debug.print("Predicted value for edge 5 at 02:30 is {}", .{edge_val});
}

// pub const Graph = struct {
//     fn init(allocator: std.mem.Allocator, vertices: std.ArrayListUnmanaged(Vertex), edge_connections: std.ArrayListUnmanaged(EdgeConnection) ) Graph {
//         _ = allocator;
//         _ = vertices;
//         _ = edge_connections;
//         // TODO: Create actual network from files
//         return Graph {};
//     }
//
//     // TODO: Make sure astar can walk this graph
// };

// // TODO: Make this return the actual route in a format that makes sense
// fn some_astar_idunno(start_node: usize, end_node: usize, lookup_table: std.AutoHashMap(Query, f32), network: *Graph) !Route {
//     _ = start_node;
//     _ = end_node;
//     _ = lookup_table;
//     _ = network;
//     // TODO: Do Astar magic here!
// }
