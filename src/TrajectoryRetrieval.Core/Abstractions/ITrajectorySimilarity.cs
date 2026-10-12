using TrajectoryRetrieval.Core.Models;

namespace TrajectoryRetrieval.Core.Abstractions;

/// <summary>
/// Similarity between a query and a batch of candidates (higher is more similar).
/// Own implementations first (e.g. Jaccard, LCSS), later replaced by the similarity group (task 4).
/// Batched so a remote implementation (e.g. a Python service) only needs one call per query.
/// </summary>
public interface ITrajectorySimilarity
{
    Task<IReadOnlyList<double>> ComputeAsync(
        Trajectory query,
        IReadOnlyList<Trajectory> candidates,
        CancellationToken cancellationToken = default);
}
