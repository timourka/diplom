using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Models.Entities;
using PostgreSQLRepository;
using ProductsDateAPI.Helpers;
using ProductsDateAPI.Models;
using System.IO.Compression;

namespace ProductsDateAPI.Controllers;

[ApiController]
[Route("api/error-reports")]
[Authorize]
public class ErrorReportsDatasetController : ControllerBase
{
    private readonly AppDbContext _db;
    private readonly IWebHostEnvironment _env;

    public ErrorReportsDatasetController(AppDbContext db, IWebHostEnvironment env)
    {
        _db = db;
        _env = env;
    }

    [HttpPost("upload-dataset")]
    [Consumes("multipart/form-data")]
    [RequestSizeLimit(400_000_000)]
    public async Task<IActionResult> UploadDataset([FromForm] UploadDatasetForm form, CancellationToken ct)
    {
        var datasetZip = form.datasetZip;
        var comment = form.comment;

        if (datasetZip is null || datasetZip.Length == 0)
            return BadRequest("datasetZip is required.");

        var userId = User.GetUserIdOrThrow();

        // 1) создаём папку
        var root = Path.Combine(_env.ContentRootPath, "uploads", "datasets");
        Directory.CreateDirectory(root);

        var folderName = $"{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";
        var datasetFolder = Path.Combine(root, folderName);
        Directory.CreateDirectory(datasetFolder);

        // 2) сохраняем zip временно
        var zipPath = Path.Combine(datasetFolder, "dataset.zip");
        await using (var fs = System.IO.File.Create(zipPath))
            await datasetZip.CopyToAsync(fs, ct);

        // 3) распаковка
        var extractFolder = Path.Combine(datasetFolder, "extracted");
        Directory.CreateDirectory(extractFolder);

        ZipFile.ExtractToDirectory(zipPath, extractFolder);

        // 4) валидация структуры
        var imagesDir = Path.Combine(extractFolder, "images");
        var labelsDir = Path.Combine(extractFolder, "labels");

        if (!Directory.Exists(imagesDir))
            return BadRequest("Zip must contain /images folder.");
        if (!Directory.Exists(labelsDir))
            return BadRequest("Zip must contain /labels folder.");

        var imagesCount = Directory.GetFiles(imagesDir, "*.*", SearchOption.TopDirectoryOnly)
            .Count(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase));

        if (imagesCount == 0)
            return BadRequest("No images found in /images.");

        // 5) записываем VideoSample, путь = папка датасета (как ты хотел)
        var vs = new VideoSample
        {
            VideoPath = Path.Combine("uploads", "datasets", folderName).Replace("\\", "/"),
            Source = "mobile_dataset"
        };
        _db.VideoSamples.Add(vs);
        await _db.SaveChangesAsync(ct);

        // 6) ErrorReport
        var report = new ErrorReport
        {
            UserId = userId,
            VideoId = vs.Id,
            Comment = string.IsNullOrWhiteSpace(comment) ? null : comment.Trim(),
            CreatedAt = DateTime.UtcNow,
            Resolved = false
        };
        _db.ErrorReports.Add(report);
        await _db.SaveChangesAsync(ct);

        return Ok(new UploadDatasetResponse(report.Id, vs.Id, vs.VideoPath));
    }
}