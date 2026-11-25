# Build context is the repository root (see docker-compose.yml)
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY frontend/src/TrajectoryRetrieval.Core/TrajectoryRetrieval.Core.csproj frontend/src/TrajectoryRetrieval.Core/
COPY frontend/src/TrajectoryRetrieval.Api/TrajectoryRetrieval.Api.csproj frontend/src/TrajectoryRetrieval.Api/
RUN dotnet restore frontend/src/TrajectoryRetrieval.Api/TrajectoryRetrieval.Api.csproj
COPY frontend/src/ frontend/src/
RUN dotnet publish frontend/src/TrajectoryRetrieval.Api/TrajectoryRetrieval.Api.csproj -c Release -o /app --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app .
EXPOSE 8080
ENTRYPOINT ["dotnet", "TrajectoryRetrieval.Api.dll"]
