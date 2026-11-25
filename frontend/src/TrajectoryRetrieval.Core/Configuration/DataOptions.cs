namespace TrajectoryRetrieval.Core.Configuration;

/// <summary>Paths to the input data. Relative paths are resolved against the API's content root.</summary>
public sealed class DataOptions
{
    public const string SectionName = "Data";

    public string VertexCsv { get; set; } = "";

    public string EdgeConnectionsCsv { get; set; } = "";

    /// <summary>Directory with the cleaned per-day travel time matrices (edge_data_day*_cleaned.csv).</summary>
    public string TravelTimeDirectory { get; set; } = "";
}
