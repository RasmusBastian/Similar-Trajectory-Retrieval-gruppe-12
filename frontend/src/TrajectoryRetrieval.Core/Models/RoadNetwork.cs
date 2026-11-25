namespace TrajectoryRetrieval.Core.Models;

/// <summary>A node in the road network (vertex.csv).</summary>
public sealed record Vertex(long NodeId, double Longitude, double Latitude);

/// <summary>A directed road segment between two vertices (edge_connections.csv).</summary>
public sealed record Edge(int EdgeId, long StartNodeId, long EndNodeId);
