using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Models.Entities;
using PostgreSQLRepository;
using ProductsDateAPI.Helpers;
using ProductsDateAPI.Models;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;

namespace ProductsDateAPI.Controllers;

[ApiController]
[Route("api/error-reports")]
[Authorize]
public class ErrorReportsDatasetController : ControllerBase
{
    private const double ValidationMinIoU = 0.20;
    private readonly AppDbContext _db;
    private readonly IWebHostEnvironment _env;
    private readonly IDataProtector _validationProtector;

    public ErrorReportsDatasetController(AppDbContext db, IWebHostEnvironment env, IDataProtectionProvider dataProtectionProvider)
    {
        _db = db;
        _env = env;
        _validationProtector = dataProtectionProvider.CreateProtector("error-report-validation-frame-v1");
    }

    [HttpGet("validation-frame")]
    public async Task<ActionResult<ReportValidationFrameResponse>> GetValidationFrame(CancellationToken ct)
    {
        var userId = User.GetUserIdOrThrow();
        var user = await _db.Users.AsNoTracking().FirstOrDefaultAsync(x => x.Id == userId, ct);
        if (user is null)
            return Unauthorized();
        if (user.IsBlocked)
            return StatusCode(StatusCodes.Status403Forbidden, "Профиль заблокирован.");

        var approvedReports = await _db.ErrorReports
            .AsNoTracking()
            .Include(x => x.Video)
            .Where(x => x.Approved && x.Video != null)
            .OrderByDescending(x => x.CreatedAt)
            .Take(50)
            .ToListAsync(ct);

        if (approvedReports.Count == 0)
            return NoContent();

        foreach (var report in approvedReports.OrderBy(_ => Random.Shared.Next()))
        {
            var datasetRoot = GetExtractedDatasetRoot(report.Video!.VideoPath);
            var imagesDir = Path.Combine(datasetRoot, "images");
            var labelsDir = Path.Combine(datasetRoot, "labels");

            if (!Directory.Exists(imagesDir) || !Directory.Exists(labelsDir))
                continue;

            var candidates = Directory.GetFiles(imagesDir, "*.*", SearchOption.TopDirectoryOnly)
                .Where(IsSupportedImage)
                .Where(IsUserReportFrame)
                .OrderBy(_ => Random.Shared.Next())
                .ToList();

            foreach (var imagePath in candidates)
            {
                var imageName = Path.GetFileName(imagePath);
                var labelPath = Path.Combine(labelsDir, Path.ChangeExtension(imageName, ".txt"));
                if (!System.IO.File.Exists(labelPath))
                    continue;

                var expected = await ReadYoloBoxesAsync(labelPath, ct);
                if (expected.Count == 0)
                    continue;

                var bytes = await System.IO.File.ReadAllBytesAsync(imagePath, ct);
                var extension = Path.GetExtension(imagePath).ToLowerInvariant();
                var contentType = extension == ".png" ? "image/png" : "image/jpeg";
                var challengeFileName = $"validation_{Guid.NewGuid():N}{(extension == ".png" ? ".png" : ".jpg")}";
                var payload = new ValidationPayload(
                    FileName: challengeFileName,
                    ExpectedBoxes: expected,
                    CreatedAtUtc: DateTime.UtcNow
                );
                var token = _validationProtector.Protect(JsonSerializer.Serialize(payload));

                return Ok(new ReportValidationFrameResponse(
                    token,
                    challengeFileName,
                    Convert.ToBase64String(bytes),
                    contentType
                ));
            }
        }

        return NoContent();
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
        var user = await _db.Users.FirstOrDefaultAsync(x => x.Id == userId, ct);
        if (user is null)
            return Unauthorized();
        if (user.IsBlocked)
            return StatusCode(StatusCodes.Status403Forbidden, "Профиль заблокирован.");

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

        var validationResult = await ValidateControlFrameIfPresentAsync(user, labelsDir, form.validationToken, form.validationFrameName, ct);
        if (validationResult is not null)
        {
            TryDeleteDirectory(datasetFolder);
            return validationResult;
        }

        RemoveValidationFrameIfPresent(imagesDir, labelsDir, form.validationFrameName);

        var imagesCount = Directory.GetFiles(imagesDir, "*.*", SearchOption.TopDirectoryOnly)
            .Where(IsSupportedImage)
            .Count(IsUserReportFrame);

        if (imagesCount == 0)
        {
            TryDeleteDirectory(datasetFolder);
            return BadRequest("No report images found in /images.");
        }

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
            Resolved = false,
            Approved = false,
            FramesCount = imagesCount
        };
        _db.ErrorReports.Add(report);
        await _db.SaveChangesAsync(ct);

        return Ok(new UploadDatasetResponse(report.Id, vs.Id, vs.VideoPath));
    }

    private async Task<IActionResult?> ValidateControlFrameIfPresentAsync(
        User user,
        string labelsDir,
        string? validationToken,
        string? validationFrameName,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(validationToken) && string.IsNullOrWhiteSpace(validationFrameName))
            return null;

        if (string.IsNullOrWhiteSpace(validationToken) || string.IsNullOrWhiteSpace(validationFrameName))
            return await BlockUserAsync(user, "Проверочный кадр повреждён: неполные данные проверки.", ct);

        ValidationPayload? payload;
        try
        {
            payload = JsonSerializer.Deserialize<ValidationPayload>(_validationProtector.Unprotect(validationToken));
        }
        catch
        {
            return await BlockUserAsync(user, "Проверочный кадр повреждён: недействительный токен проверки.", ct);
        }

        if (payload is null || payload.ExpectedBoxes.Count == 0)
            return await BlockUserAsync(user, "Проверочный кадр повреждён: нет эталонной разметки.", ct);

        if ((DateTime.UtcNow - payload.CreatedAtUtc).TotalHours > 24)
            return await BlockUserAsync(user, "Проверочный кадр устарел. Создай отчёт заново.", ct);

        var safeFrameName = Path.GetFileName(validationFrameName);
        if (!string.Equals(safeFrameName, payload.FileName, StringComparison.OrdinalIgnoreCase))
            return await BlockUserAsync(user, "Проверочный кадр повреждён: имя кадра не совпадает.", ct);

        var labelPath = Path.Combine(labelsDir, Path.ChangeExtension(safeFrameName, ".txt"));
        if (!System.IO.File.Exists(labelPath))
            return await BlockUserAsync(user, "Проверочный кадр был пропущен или не размечен.", ct);

        var actual = await ReadYoloBoxesAsync(labelPath, ct);
        if (actual.Count == 0)
            return await BlockUserAsync(user, "Проверочный кадр не содержит разметки.", ct);

        var bestIoU = actual
            .SelectMany(a => payload.ExpectedBoxes.Select(e => CalculateIoU(a, e)))
            .DefaultIfEmpty(0)
            .Max();

        if (bestIoU < ValidationMinIoU)
            return await BlockUserAsync(user, $"Проверка разметки не пройдена. Аккаунт заблокирован.", ct);

        return null;
    }

    private async Task<IActionResult> BlockUserAsync(User user, string message, CancellationToken ct)
    {
        user.IsBlocked = true;
        await _db.SaveChangesAsync(ct);
        return StatusCode(StatusCodes.Status403Forbidden, message);
    }

    private string GetExtractedDatasetRoot(string videoPath) => Path.Combine(
        _env.ContentRootPath,
        videoPath.Replace("/", Path.DirectorySeparatorChar.ToString()),
        "extracted"
    );

    private static bool IsSupportedImage(string path) =>
        path.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
        path.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
        path.EndsWith(".png", StringComparison.OrdinalIgnoreCase);

    private static bool IsUserReportFrame(string path) =>
        !Path.GetFileName(path).StartsWith("validation_", StringComparison.OrdinalIgnoreCase);

    private static void RemoveValidationFrameIfPresent(string imagesDir, string labelsDir, string? validationFrameName)
    {
        if (string.IsNullOrWhiteSpace(validationFrameName))
            return;

        var safeName = Path.GetFileName(validationFrameName);
        if (string.IsNullOrWhiteSpace(safeName) ||
            !safeName.StartsWith("validation_", StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        TryDeleteFile(Path.Combine(imagesDir, safeName));
        TryDeleteFile(Path.Combine(labelsDir, Path.ChangeExtension(safeName, ".txt")));
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (System.IO.File.Exists(path))
                System.IO.File.Delete(path);
        }
        catch
        {
        }
    }

    private static async Task<List<YoloBboxDto>> ReadYoloBoxesAsync(string labelPath, CancellationToken ct)
    {
        var lines = await System.IO.File.ReadAllLinesAsync(labelPath, ct);
        var bboxes = new List<YoloBboxDto>();

        foreach (var line in lines.Where(x => !string.IsNullOrWhiteSpace(x)))
        {
            var parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 5)
                continue;

            if (!int.TryParse(parts[0], out var classId))
                continue;
            if (!double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var xc))
                continue;
            if (!double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var yc))
                continue;
            if (!double.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out var w))
                continue;
            if (!double.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out var h))
                continue;

            bboxes.Add(new YoloBboxDto(classId, xc, yc, w, h));
        }

        return bboxes;
    }

    private static double CalculateIoU(YoloBboxDto a, YoloBboxDto b)
    {
        var ar = ToRect(a);
        var br = ToRect(b);

        var left = Math.Max(ar.Left, br.Left);
        var top = Math.Max(ar.Top, br.Top);
        var right = Math.Min(ar.Right, br.Right);
        var bottom = Math.Min(ar.Bottom, br.Bottom);

        var intersectionW = Math.Max(0, right - left);
        var intersectionH = Math.Max(0, bottom - top);
        var intersection = intersectionW * intersectionH;

        var areaA = Math.Max(0, ar.Right - ar.Left) * Math.Max(0, ar.Bottom - ar.Top);
        var areaB = Math.Max(0, br.Right - br.Left) * Math.Max(0, br.Bottom - br.Top);
        var union = areaA + areaB - intersection;

        return union <= 0 ? 0 : intersection / union;
    }

    private static NormalizedRect ToRect(YoloBboxDto box)
    {
        var left = box.Xc - box.W / 2.0;
        var top = box.Yc - box.H / 2.0;
        var right = box.Xc + box.W / 2.0;
        var bottom = box.Yc + box.H / 2.0;
        return new NormalizedRect(left, top, right, bottom);
    }

    private static void TryDeleteDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path))
                Directory.Delete(path, recursive: true);
        }
        catch
        {
        }
    }

    private sealed record ValidationPayload(string FileName, List<YoloBboxDto> ExpectedBoxes, DateTime CreatedAtUtc);
    private sealed record NormalizedRect(double Left, double Top, double Right, double Bottom);
}
