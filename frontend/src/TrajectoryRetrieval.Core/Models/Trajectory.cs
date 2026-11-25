namespace TrajectoryRetrieval.Core.Models;

/// <summary>A raw GPS measurement before map matching.</summary>
public sealed record GpsPoint(double Latitude, double Longitude, DateTimeOffset Timestamp);

/// <summary>A map-matched trajectory, represented as the ordered sequence of road segments it traversed.</summary>
public sealed record Trajectory(
    string TrajectoryId,
    IReadOnlyList<int> EdgeIds,
    DateTimeOffset? StartTime = null,
    DateTimeOffset? EndTime = null);

/// <summary>A retrieved trajectory together with its similarity to the query.</summary>
public sealed record SimilarTrajectory(Trajectory Trajectory, double Score);
