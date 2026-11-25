using TrajectoryRetrieval.Core.Configuration;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenApi();
builder.Services.Configure<DataOptions>(builder.Configuration.GetSection(DataOptions.SectionName));

// Register implementations of the Core abstractions here, e.g.
// builder.Services.AddSingleton<ITrajectorySimilarity, JaccardSimilarity>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.MapGet("/health", () => Results.Ok(new { status = "ok" }))
    .WithName("Health");

app.Run();
