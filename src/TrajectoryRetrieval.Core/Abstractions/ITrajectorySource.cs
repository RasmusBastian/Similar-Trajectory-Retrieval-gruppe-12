using TrajectoryRetrieval.Core.Models;

namespace TrajectoryRetrieval.Core.Abstractions;

/// <summary>
/// Historical map-matched trajectories.
/// Own implementation first (synthetic data), later replaced by the map matching group's (task 1) data.
/// </summary>
public interface ITrajectorySource
{
    IAsyncEnumerable<Trajectory> GetAllAsync(CancellationToken cancellationToken = default);

    Task<Trajectory?> GetByIdAsync(string trajectoryId, CancellationToken cancellationToken = default);
}
