using Microsoft.Extensions.Options;
using ProductsDateAPI.Models;

namespace ProductsDateAPI.Services;

public class TrainingFileStorage
{
    private readonly IWebHostEnvironment _env;
    private readonly TrainingServiceOptions _options;

    public TrainingFileStorage(IWebHostEnvironment env, IOptions<TrainingServiceOptions> options)
    {
        _env = env;
        _options = options.Value;
    }

    public string RootPath
    {
        get
        {
            var root = string.IsNullOrWhiteSpace(_options.StorageRoot)
                ? "storage/training"
                : _options.StorageRoot;

            return Path.IsPathRooted(root)
                ? root
                : Path.Combine(_env.ContentRootPath, root);
        }
    }

    public string SaveDatasetZip(string jobId, string sourceZipPath)
    {
        var jobDir = Path.Combine(RootPath, "jobs", jobId);
        Directory.CreateDirectory(jobDir);

        var destination = Path.Combine(jobDir, "dataset.zip");
        File.Copy(sourceZipPath, destination, overwrite: true);
        return ToRelativePath(destination);
    }

    public async Task<string> SaveArtifactAsync(string jobId, IFormFile file, string fileName, CancellationToken ct)
    {
        var safeName = MakeSafeFileName(fileName);
        var artifactDir = Path.Combine(RootPath, "models", jobId);
        Directory.CreateDirectory(artifactDir);

        var destination = Path.Combine(artifactDir, safeName);
        await using var stream = File.Create(destination);
        await file.CopyToAsync(stream, ct);
        return ToRelativePath(destination);
    }

    public string ToAbsolutePath(string relativePath)
    {
        var normalized = relativePath.Replace('/', Path.DirectorySeparatorChar);
        return Path.IsPathRooted(normalized) ? normalized : Path.Combine(RootPath, normalized);
    }

    public FileStream OpenRead(string relativePath)
        => File.OpenRead(ToAbsolutePath(relativePath));

    public void TryDeleteFile(string? relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
            return;

        try
        {
            var absolute = ToAbsolutePath(relativePath);
            if (File.Exists(absolute))
                File.Delete(absolute);
        }
        catch
        {
            // best effort cleanup
        }
    }

    private string ToRelativePath(string absolutePath)
        => Path.GetRelativePath(RootPath, absolutePath).Replace("\\", "/");

    private static string MakeSafeFileName(string fileName)
    {
        var cleaned = string.Join("_", Path.GetFileName(fileName).Split(Path.GetInvalidFileNameChars(), StringSplitOptions.RemoveEmptyEntries)).Trim();
        return string.IsNullOrWhiteSpace(cleaned) ? $"artifact_{Guid.NewGuid():N}.bin" : cleaned;
    }
}
