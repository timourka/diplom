# syntax=docker/dockerfile:1

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
EXPOSE 8080
ENV ASPNETCORE_URLS=http://+:8080

RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /app/storage /app/temp /app/dataset /app/uploads \
    && chown -R appuser:appuser /app

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY ["ProductsDateAPI/ProductsDateAPI.csproj", "ProductsDateAPI/"]
COPY ["Contracts/Contracts.csproj", "Contracts/"]
COPY ["Models/Models.csproj", "Models/"]
COPY ["PostgreSQLRepository/PostgreSQLRepository.csproj", "PostgreSQLRepository/"]
RUN dotnet restore "ProductsDateAPI/ProductsDateAPI.csproj"

COPY . .
RUN dotnet publish "ProductsDateAPI/ProductsDateAPI.csproj" \
    -c Release \
    -o /app/publish \
    --no-restore

FROM runtime AS final
COPY --from=build /app/publish .
USER appuser
ENTRYPOINT ["dotnet", "ProductsDateAPI.dll"]
