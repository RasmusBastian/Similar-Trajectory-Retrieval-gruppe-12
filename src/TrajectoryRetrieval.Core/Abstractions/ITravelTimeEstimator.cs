using TrajectoryRetrieval.Core.Models;

namespace TrajectoryRetrieval.Core.Abstractions;

/// <summary>
/// Estimated travel time of a trajectory at a given departure time.
/// Own implementation first (average edge times from legacy data), later replaced by the travel time group (task 3).
/// </summary>
public interface ITravelTimeEstimator
{
    Task<TimeSpan> EstimateAsync(Trajectory trajectory, DateTimeOffset departure, CancellationToken cancellationToken = default);
}
