using System.IO.Compression;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using Models.Entities;
using PostgreSQLRepository;

namespace ProductsDateAPI.Services;

public sealed class BackupService
{
    private const string BackupFormat = "productsdate-backup";
    private const int BackupVersion = 1;

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true
    };

    private readonly AppDbContext _db;
    private readonly IWebHostEnvironment _env;
    private readonly TrainingFileStorage _trainingStorage;

    public BackupService(AppDbContext db, IWebHostEnvironment env, TrainingFileStorage trainingStorage)
    {
        _db = db;
        _env = env;
        _trainingStorage = trainingStorage;
    }

    public async Task<string> CreateBackupZipAsync(CancellationToken ct)
    {
        var backupDir = Path.Combine(Path.GetTempPath(), "productsdate-backups");
        Directory.CreateDirectory(backupDir);

        var zipPath = Path.Combine(backupDir, $"productsdate_backup_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}.zip");

        await using var zipStream = File.Create(zipPath);
        using var archive = new ZipArchive(zipStream, ZipArchiveMode.Create, leaveOpen: false);

        await AddJsonEntryAsync(archive, "manifest.json", new BackupManifest(
            BackupFormat,
            BackupVersion,
            DateTime.UtcNow,
            "ProductsDateAPI"), ct);

        var snapshot = await CreateDatabaseSnapshotAsync(ct);
        await AddJsonEntryAsync(archive, "database.json", snapshot, ct);

        AddDirectoryToArchive(archive, _trainingStorage.RootPath, "storage/training");
        AddDirectoryToArchive(archive, UploadsRootPath, "uploads");

        return zipPath;
    }

    public async Task<BackupImportResult> ImportBackupZipAsync(IFormFile backupZip, bool replaceExisting, CancellationToken ct)
    {
        if (backupZip.Length == 0)
            throw new InvalidOperationException("ZIP-файл пустой.");

        var tempRoot = Path.Combine(Path.GetTempPath(), "productsdate-backup-import", Guid.NewGuid().ToString("N"));
        var zipPath = Path.Combine(tempRoot, "backup.zip");
        var extractRoot = Path.Combine(tempRoot, "extracted");

        Directory.CreateDirectory(tempRoot);
        Directory.CreateDirectory(extractRoot);

        try
        {
            await using (var stream = File.Create(zipPath))
                await backupZip.CopyToAsync(stream, ct);

            ExtractZipSafely(zipPath, extractRoot);

            var manifest = await ReadJsonFileAsync<BackupManifest>(Path.Combine(extractRoot, "manifest.json"), ct);
            if (!string.Equals(manifest.Format, BackupFormat, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Неверный формат backup-архива.");
            if (manifest.Version > BackupVersion)
                throw new InvalidOperationException($"Backup создан в более новой версии формата: {manifest.Version}.");

            var snapshot = await ReadJsonFileAsync<BackupDatabaseSnapshot>(Path.Combine(extractRoot, "database.json"), ct);

            if (!replaceExisting && await DatabaseHasDataAsync(ct))
            {
                throw new InvalidOperationException(
                    "В базе уже есть данные. Для полного восстановления передайте replaceExisting=true.");
            }

            await using var tx = await _db.Database.BeginTransactionAsync(ct);

            if (replaceExisting)
            {
                await ClearDatabaseAsync(ct);
                DeleteDirectoryIfExists(_trainingStorage.RootPath);
                DeleteDirectoryIfExists(UploadsRootPath);
            }

            CopyDirectoryContentsIfExists(Path.Combine(extractRoot, "storage", "training"), _trainingStorage.RootPath);
            CopyDirectoryContentsIfExists(Path.Combine(extractRoot, "uploads"), UploadsRootPath);

            RestoreDatabaseSnapshot(snapshot);
            await _db.SaveChangesAsync(ct);
            await ResetSequencesAsync(ct);

            await tx.CommitAsync(ct);

            return new BackupImportResult(
                snapshot.Users.Count,
                snapshot.Products.Count,
                snapshot.StoredProducts.Count,
                snapshot.VideoSamples.Count,
                snapshot.ErrorReports.Count,
                snapshot.ModelVersions.Count,
                snapshot.TrainingJobs.Count,
                replaceExisting);
        }
        finally
        {
            try
            {
                if (Directory.Exists(tempRoot))
                    Directory.Delete(tempRoot, recursive: true);
            }
            catch
            {
                // best effort cleanup
            }
        }
    }

    private string UploadsRootPath => Path.Combine(_env.ContentRootPath, "uploads");

    private async Task<BackupDatabaseSnapshot> CreateDatabaseSnapshotAsync(CancellationToken ct)
    {
        return new BackupDatabaseSnapshot
        {
            Users = await _db.Users.AsNoTracking()
                .OrderBy(x => x.Id)
                .Select(x => new UserBackupDto(x.Id, x.Email, x.PasswordHash, x.IsBlocked, x.SettingsJson, x.CreatedAt))
                .ToListAsync(ct),

            Products = await _db.Products.AsNoTracking()
                .OrderBy(x => x.Id)
                .Select(x => new ProductBackupDto(x.Id, x.Name, x.Manufacturer, x.Barcode))
                .ToListAsync(ct),

            StoredProducts = await _db.StoredProducts.AsNoTracking()
                .OrderBy(x => x.Id)
                .Select(x => new StoredProductBackupDto(x.Id, x.UserId, x.ProductId, x.ManufactureAt, x.ExpiryAt, x.CreatedAt))
                .ToListAsync(ct),

            VideoSamples = await _db.VideoSamples.AsNoTracking()
                .OrderBy(x => x.Id)
                .Select(x => new VideoSampleBackupDto(x.Id, x.VideoPath, x.Source))
                .ToListAsync(ct),

            ErrorReports = await _db.ErrorReports.AsNoTracking()
                .OrderBy(x => x.Id)
                .Select(x => new ErrorReportBackupDto(x.Id, x.UserId, x.VideoId, x.ModelVersionId, x.Comment, x.CreatedAt, x.Resolved, x.Approved, x.FramesCount))
                .ToListAsync(ct),

            ModelVersions = await _db.ModelVersions.AsNoTracking()
                .OrderBy(x => x.Id)
                .Select(x => new ModelVersionBackupDto(
                    x.Id,
                    x.TrainedAt,
                    x.MetricsJson,
                    x.ExternalJobId,
                    x.BaseModel,
                    x.BestWeightsPath,
                    x.MobileModelPath,
                    x.MobileFormat,
                    x.MobileModelFileName,
                    x.MobileModelContentType,
                    x.IsPublished,
                    x.IsPinned,
                    x.IsDeleted,
                    x.DeletedAt))
                .ToListAsync(ct),

            TrainingJobs = await _db.TrainingJobs.AsNoTracking()
                .OrderBy(x => x.Id)
                .Select(x => new TrainingJobBackupDto(
                    x.Id,
                    x.JobId,
                    x.Status,
                    x.Message,
                    x.CreatedAt,
                    x.StartedAt,
                    x.FinishedAt,
                    x.AssignedAt,
                    x.HeartbeatAt,
                    x.ImagesCount,
                    x.ClientId,
                    x.DatasetZipPath,
                    x.BaseModel,
                    x.Epochs,
                    x.ImgSize,
                    x.Batch,
                    x.Device,
                    x.ExportInt8,
                    x.ExportNms,
                    x.MobileFormat,
                    x.QuantizationFraction,
                    x.BestWeightsPath,
                    x.MobileModelPath,
                    x.MobileModelFileName,
                    x.MobileModelContentType,
                    x.MetricsJson,
                    x.CancellationRequested))
                .ToListAsync(ct)
        };
    }

    private async Task<bool> DatabaseHasDataAsync(CancellationToken ct)
    {
        return await _db.Users.AnyAsync(ct)
               || await _db.Products.AnyAsync(ct)
               || await _db.StoredProducts.AnyAsync(ct)
               || await _db.VideoSamples.AnyAsync(ct)
               || await _db.ErrorReports.AnyAsync(ct)
               || await _db.ModelVersions.AnyAsync(ct)
               || await _db.TrainingJobs.AnyAsync(ct);
    }

    private async Task ClearDatabaseAsync(CancellationToken ct)
    {
        await _db.ErrorReports.ExecuteDeleteAsync(ct);
        await _db.StoredProducts.ExecuteDeleteAsync(ct);
        await _db.TrainingJobs.ExecuteDeleteAsync(ct);
        await _db.VideoSamples.ExecuteDeleteAsync(ct);
        await _db.ModelVersions.ExecuteDeleteAsync(ct);
        await _db.Products.ExecuteDeleteAsync(ct);
        await _db.Users.ExecuteDeleteAsync(ct);
    }

    private void RestoreDatabaseSnapshot(BackupDatabaseSnapshot snapshot)
    {
        _db.Users.AddRange(snapshot.Users.Select(x => new User
        {
            Id = x.Id,
            Email = x.Email,
            PasswordHash = x.PasswordHash,
            IsBlocked = x.IsBlocked,
            SettingsJson = x.SettingsJson,
            CreatedAt = x.CreatedAt
        }));

        _db.Products.AddRange(snapshot.Products.Select(x => new Product
        {
            Id = x.Id,
            Name = x.Name,
            Manufacturer = x.Manufacturer,
            Barcode = x.Barcode
        }));

        _db.VideoSamples.AddRange(snapshot.VideoSamples.Select(x => new VideoSample
        {
            Id = x.Id,
            VideoPath = x.VideoPath,
            Source = x.Source
        }));

        _db.ModelVersions.AddRange(snapshot.ModelVersions.Select(x => new ModelVersion
        {
            Id = x.Id,
            TrainedAt = x.TrainedAt,
            MetricsJson = x.MetricsJson,
            ExternalJobId = x.ExternalJobId,
            BaseModel = x.BaseModel,
            BestWeightsPath = x.BestWeightsPath,
            MobileModelPath = x.MobileModelPath,
            MobileFormat = x.MobileFormat,
            MobileModelFileName = x.MobileModelFileName,
            MobileModelContentType = x.MobileModelContentType,
            IsPublished = x.IsPublished,
            IsPinned = x.IsPinned,
            IsDeleted = x.IsDeleted,
            DeletedAt = x.DeletedAt
        }));

        _db.StoredProducts.AddRange(snapshot.StoredProducts.Select(x => new StoredProduct
        {
            Id = x.Id,
            UserId = x.UserId,
            ProductId = x.ProductId,
            ManufactureAt = x.ManufactureAt,
            ExpiryAt = x.ExpiryAt,
            CreatedAt = x.CreatedAt
        }));

        _db.TrainingJobs.AddRange(snapshot.TrainingJobs.Select(x => new TrainingJob
        {
            Id = x.Id,
            JobId = x.JobId,
            Status = x.Status,
            Message = x.Message,
            CreatedAt = x.CreatedAt,
            StartedAt = x.StartedAt,
            FinishedAt = x.FinishedAt,
            AssignedAt = x.AssignedAt,
            HeartbeatAt = x.HeartbeatAt,
            ImagesCount = x.ImagesCount,
            ClientId = x.ClientId,
            DatasetZipPath = x.DatasetZipPath,
            BaseModel = x.BaseModel,
            Epochs = x.Epochs,
            ImgSize = x.ImgSize,
            Batch = x.Batch,
            Device = x.Device,
            ExportInt8 = x.ExportInt8,
            ExportNms = x.ExportNms,
            MobileFormat = x.MobileFormat,
            QuantizationFraction = x.QuantizationFraction,
            BestWeightsPath = x.BestWeightsPath,
            MobileModelPath = x.MobileModelPath,
            MobileModelFileName = x.MobileModelFileName,
            MobileModelContentType = x.MobileModelContentType,
            MetricsJson = x.MetricsJson,
            CancellationRequested = x.CancellationRequested
        }));

        _db.ErrorReports.AddRange(snapshot.ErrorReports.Select(x => new ErrorReport
        {
            Id = x.Id,
            UserId = x.UserId,
            VideoId = x.VideoId,
            ModelVersionId = x.ModelVersionId,
            Comment = x.Comment,
            CreatedAt = x.CreatedAt,
            Resolved = x.Resolved,
            Approved = x.Approved,
            FramesCount = x.FramesCount
        }));
    }

    private async Task ResetSequencesAsync(CancellationToken ct)
    {
        await ResetSequenceAsync("Users", ct);
        await ResetSequenceAsync("Products", ct);
        await ResetSequenceAsync("StoredProducts", ct);
        await ResetSequenceAsync("VideoSamples", ct);
        await ResetSequenceAsync("ErrorReports", ct);
        await ResetSequenceAsync("ModelVersions", ct);
        await ResetSequenceAsync("TrainingJobs", ct);
    }

    private async Task ResetSequenceAsync(string tableName, CancellationToken ct)
    {
        var quotedTable = tableName.Replace("\"", "\"\"");
        var sql = $"""
                  SELECT setval(
                      pg_get_serial_sequence('"{quotedTable}"', 'Id'),
                      GREATEST(COALESCE((SELECT MAX("Id") FROM "{quotedTable}"), 1), 1),
                      EXISTS(SELECT 1 FROM "{quotedTable}")
                  );
                  """;
        await _db.Database.ExecuteSqlRawAsync(sql, cancellationToken: ct);
    }

    private static async Task AddJsonEntryAsync<T>(ZipArchive archive, string entryName, T value, CancellationToken ct)
    {
        var entry = archive.CreateEntry(entryName, CompressionLevel.Optimal);
        await using var stream = entry.Open();
        await JsonSerializer.SerializeAsync(stream, value, JsonOptions, ct);
    }

    private static async Task<T> ReadJsonFileAsync<T>(string path, CancellationToken ct)
    {
        if (!File.Exists(path))
            throw new InvalidOperationException($"В backup-архиве нет обязательного файла {Path.GetFileName(path)}.");

        await using var stream = File.OpenRead(path);
        return await JsonSerializer.DeserializeAsync<T>(stream, JsonOptions, ct)
               ?? throw new InvalidOperationException($"Не удалось прочитать {Path.GetFileName(path)}.");
    }

    private static void AddDirectoryToArchive(ZipArchive archive, string sourceDirectory, string entryPrefix)
    {
        if (!Directory.Exists(sourceDirectory))
            return;

        var sourceRoot = Path.GetFullPath(sourceDirectory);
        foreach (var file in Directory.EnumerateFiles(sourceRoot, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(sourceRoot, file).Replace('\\', '/');
            var entryName = JoinZipPath(entryPrefix, relative);
            archive.CreateEntryFromFile(file, entryName, CompressionLevel.Fastest);
        }
    }

    private static void ExtractZipSafely(string zipPath, string destinationDirectory)
    {
        var destinationRoot = Path.GetFullPath(destinationDirectory);
        Directory.CreateDirectory(destinationRoot);

        using var archive = ZipFile.OpenRead(zipPath);
        foreach (var entry in archive.Entries)
        {
            if (string.IsNullOrWhiteSpace(entry.FullName))
                continue;

            var destinationPath = Path.GetFullPath(Path.Combine(destinationRoot, entry.FullName));
            if (!destinationPath.StartsWith(destinationRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
                && !string.Equals(destinationPath, destinationRoot, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("Backup-архив содержит небезопасные пути файлов.");
            }

            if (entry.FullName.EndsWith("/", StringComparison.Ordinal))
            {
                Directory.CreateDirectory(destinationPath);
                continue;
            }

            var directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(directory))
                Directory.CreateDirectory(directory);

            entry.ExtractToFile(destinationPath, overwrite: true);
        }
    }

    private static void CopyDirectoryContentsIfExists(string sourceDirectory, string destinationDirectory)
    {
        if (!Directory.Exists(sourceDirectory))
            return;

        foreach (var sourcePath in Directory.EnumerateFileSystemEntries(sourceDirectory, "*", SearchOption.AllDirectories))
        {
            var relative = Path.GetRelativePath(sourceDirectory, sourcePath);
            var destinationPath = Path.Combine(destinationDirectory, relative);

            if (Directory.Exists(sourcePath))
            {
                Directory.CreateDirectory(destinationPath);
                continue;
            }

            var destinationDir = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(destinationDir))
                Directory.CreateDirectory(destinationDir);

            File.Copy(sourcePath, destinationPath, overwrite: true);
        }
    }

    private static void DeleteDirectoryIfExists(string directory)
    {
        if (Directory.Exists(directory))
            Directory.Delete(directory, recursive: true);

        Directory.CreateDirectory(directory);
    }

    private static string JoinZipPath(string left, string right)
        => $"{left.TrimEnd('/', '\\')}/{right.TrimStart('/', '\\')}".Replace('\\', '/');
}

public sealed record BackupManifest(
    string Format,
    int Version,
    DateTime CreatedAtUtc,
    string Application);

public sealed class BackupDatabaseSnapshot
{
    public List<UserBackupDto> Users { get; set; } = new();
    public List<ProductBackupDto> Products { get; set; } = new();
    public List<StoredProductBackupDto> StoredProducts { get; set; } = new();
    public List<VideoSampleBackupDto> VideoSamples { get; set; } = new();
    public List<ErrorReportBackupDto> ErrorReports { get; set; } = new();
    public List<ModelVersionBackupDto> ModelVersions { get; set; } = new();
    public List<TrainingJobBackupDto> TrainingJobs { get; set; } = new();
}

public sealed record UserBackupDto(
    int Id,
    string Email,
    string PasswordHash,
    bool IsBlocked,
    string? SettingsJson,
    DateTime CreatedAt);

public sealed record ProductBackupDto(
    int Id,
    string Name,
    string? Manufacturer,
    string? Barcode);

public sealed record StoredProductBackupDto(
    int Id,
    int UserId,
    int ProductId,
    DateTime? ManufactureAt,
    DateTime? ExpiryAt,
    DateTime CreatedAt);

public sealed record VideoSampleBackupDto(
    int Id,
    string VideoPath,
    string? Source);

public sealed record ErrorReportBackupDto(
    int Id,
    int UserId,
    int? VideoId,
    int? ModelVersionId,
    string? Comment,
    DateTime CreatedAt,
    bool Resolved,
    bool Approved,
    int FramesCount);

public sealed record ModelVersionBackupDto(
    int Id,
    DateTime TrainedAt,
    string? MetricsJson,
    string? ExternalJobId,
    string? BaseModel,
    string? BestWeightsPath,
    string? MobileModelPath,
    string? MobileFormat,
    string? MobileModelFileName,
    string? MobileModelContentType,
    bool IsPublished,
    bool IsPinned,
    bool IsDeleted,
    DateTime? DeletedAt);

public sealed record TrainingJobBackupDto(
    int Id,
    string JobId,
    string Status,
    string? Message,
    DateTime CreatedAt,
    DateTime? StartedAt,
    DateTime? FinishedAt,
    DateTime? AssignedAt,
    DateTime? HeartbeatAt,
    int ImagesCount,
    string? ClientId,
    string DatasetZipPath,
    string? BaseModel,
    int Epochs,
    int ImgSize,
    int Batch,
    string? Device,
    bool ExportInt8,
    bool ExportNms,
    string? MobileFormat,
    double QuantizationFraction,
    string? BestWeightsPath,
    string? MobileModelPath,
    string? MobileModelFileName,
    string? MobileModelContentType,
    string? MetricsJson,
    bool CancellationRequested);

public sealed record BackupImportResult(
    int Users,
    int Products,
    int StoredProducts,
    int VideoSamples,
    int ErrorReports,
    int ModelVersions,
    int TrainingJobs,
    bool ReplacedExisting);
