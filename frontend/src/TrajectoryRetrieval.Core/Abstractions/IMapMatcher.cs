using TrajectoryRetrieval.Core.Models;

namespace TrajectoryRetrieval.Core.Abstractions;

/// <summary>
/// Turns raw GPS points into a road-segment trajectory.
/// Own naive implementation first, later replaced by the map matching group (task 1).
/// </summary>
public interface IMapMatcher
{
    Task<Trajectory> MatchAsync(IReadOnlyList<GpsPoint> points, CancellationToken cancellationToken = default);
}
