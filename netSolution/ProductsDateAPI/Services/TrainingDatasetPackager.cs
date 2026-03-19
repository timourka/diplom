using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using PostgreSQLRepository;
using ProductsDateAPI.Models;
using System.IO.Compression;

namespace ProductsDateAPI.Services;

public sealed record TrainingDatasetPackage(string WorkingDirectory, string ZipPath, int ImagesCount);

public class TrainingDatasetPackager
{
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png"
    };

    private readonly AppDbContext _db;
    private readonly IWebHostEnvironment _env;
    private readonly TrainingServiceOptions _options;

    public TrainingDatasetPackager(
        AppDbContext db,
        IWebHostEnvironment env,
        IOptions<TrainingServiceOptions> options)
    {
        _db = db;
        _env = env;
        _options = options.Value;
    }

    public async Task<TrainingDatasetPackage> CreateApprovedReportsZipAsync(CancellationToken ct = default)
    {
        var reports = await _db.ErrorReports
            .AsNoTracking()
            .Include(x => x.Video)
            .Where(x => x.Approved && x.Video != null)
            .OrderBy(x => x.Id)
            .ToListAsync(ct);

        var tempRoot = ResolvePath(_options.TempRoot);
        Directory.CreateDirectory(tempRoot);

        var workDir = Path.Combine(tempRoot, $"train_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}");
        var datasetRoot = Path.Combine(workDir, "dataset");
        var imagesOut = Path.Combine(datasetRoot, "images");
        var labelsOut = Path.Combine(datasetRoot, "labels");
        Directory.CreateDirectory(imagesOut);
        Directory.CreateDirectory(labelsOut);

        var copied = 0;

        var seedDatasetPath = ResolveOptionalPath(_options.SeedDatasetPath);
        if (!string.IsNullOrWhiteSpace(seedDatasetPath) && Directory.Exists(seedDatasetPath))
        {
            copied += CopyPairs(seedDatasetPath, imagesOut, labelsOut, "seed");
        }

        foreach (var report in reports)
        {
            if (report.Video is null || string.IsNullOrWhiteSpace(report.Video.VideoPath))
                continue;

            var extractedRoot = Path.Combine(
                _env.ContentRootPath,
                report.Video.VideoPath.Replace("/", Path.DirectorySeparatorChar.ToString()),
                "extracted");

            if (!Directory.Exists(extractedRoot))
                continue;

            copied += CopyPairs(extractedRoot, imagesOut, labelsOut, $"report_{report.Id}");
        }

        if (copied == 0)
            throw new InvalidOperationException(
                "Не найдено ни одного размеченного кадра. Нужны approved отчёты с images/labels или seed dataset.");

        var yamlPath = Path.Combine(datasetRoot, "dataset.yaml");
        await File.WriteAllTextAsync(yamlPath, BuildDatasetYaml(), ct);

        var zipPath = Path.Combine(workDir, "dataset.zip");
        ZipFile.CreateFromDirectory(datasetRoot, zipPath);

        return new TrainingDatasetPackage(workDir, zipPath, copied);
    }

    private static int CopyPairs(string sourceRoot, string imagesOut, string labelsOut, string prefix)
    {
        var imagesDir = Path.Combine(sourceRoot, "images");
        var labelsDir = Path.Combine(sourceRoot, "labels");

        if (!Directory.Exists(imagesDir) || !Directory.Exists(labelsDir))
            return 0;

        var copied = 0;

        foreach (var imagePath in Directory.EnumerateFiles(imagesDir, "*.*", SearchOption.TopDirectoryOnly)
                     .Where(x => ImageExtensions.Contains(Path.GetExtension(x))))
        {
            var labelPath = Path.Combine(labelsDir, Path.GetFileNameWithoutExtension(imagePath) + ".txt");
            if (!File.Exists(labelPath))
                continue;

            copied++;
            var fileStem = $"{prefix}_{copied:D6}";
            var imageExt = Path.GetExtension(imagePath).ToLowerInvariant();

            File.Copy(imagePath, Path.Combine(imagesOut, fileStem + imageExt), overwrite: true);
            File.Copy(labelPath, Path.Combine(labelsOut, fileStem + ".txt"), overwrite: true);
        }

        return copied;
    }

    private string ResolvePath(string path)
        => Path.IsPathRooted(path) ? path : Path.Combine(_env.ContentRootPath, path);

    private string? ResolveOptionalPath(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return null;

        return ResolvePath(path);
    }

    private static string BuildDatasetYaml() =>
        "path: .\n" +
        "train: images\n" +
        "val: images\n\n" +
        "nc: 1\n" +
        "names: [expiry_date]\n";
}
