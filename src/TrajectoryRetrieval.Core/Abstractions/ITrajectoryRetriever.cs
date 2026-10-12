using TrajectoryRetrieval.Core.Models;

namespace TrajectoryRetrieval.Core.Abstractions;

/// <summary>
/// Task 5: retrieve the Top-K most similar historical trajectories for a query.
/// This is our own core logic and is not replaced by other groups.
/// </summary>
public interface ITrajectoryRetriever
{
    Task<IReadOnlyList<SimilarTrajectory>> FindMostSimilarAsync(
        Trajectory query,
        int k,
        CancellationToken cancellationToken = default);
}
