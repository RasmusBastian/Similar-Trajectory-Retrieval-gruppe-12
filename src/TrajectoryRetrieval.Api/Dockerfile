# Build context is the repository root (see docker-compose.yml)
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY src/TrajectoryRetrieval.Core/TrajectoryRetrieval.Core.csproj src/TrajectoryRetrieval.Core/
COPY src/TrajectoryRetrieval.Api/TrajectoryRetrieval.Api.csproj src/TrajectoryRetrieval.Api/
RUN dotnet restore src/TrajectoryRetrieval.Api/TrajectoryRetrieval.Api.csproj
COPY src/ src/
RUN dotnet publish src/TrajectoryRetrieval.Api/TrajectoryRetrieval.Api.csproj -c Release -o /app --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app .
EXPOSE 8080
ENTRYPOINT ["dotnet", "TrajectoryRetrieval.Api.dll"]
