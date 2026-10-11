using TrajectoryRetrieval.Core.Models;

namespace TrajectoryRetrieval.Tests;

public class TrajectoryTests
{
    [Fact]
    public void Trajectory_keeps_edge_order()
    {
        var trajectory = new Trajectory("t1", [1, 2, 4, 6]);

        Assert.Equal([1, 2, 4, 6], trajectory.EdgeIds);
    }
}
