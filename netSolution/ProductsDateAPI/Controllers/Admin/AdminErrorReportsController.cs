using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PostgreSQLRepository;
using System.Globalization;

namespace ProductsDateAPI.Controllers.Admin;

[ApiController]
[Route("api/admin/error-reports")]
[Authorize(Policy = "AdminOnly")]
public class AdminErrorReportsController : ControllerBase
{
    private readonly AppDbContext _db;
    private readonly IWebHostEnvironment _env;

    public AdminErrorReportsController(AppDbContext db, IWebHostEnvironment env)
    {
        _db = db;
        _env = env;
    }

    [HttpGet]
    public async Task<ActionResult<List<AdminErrorReportListItemDto>>> GetList(
        [FromQuery] bool? onlyApproved,
        [FromQuery] int skip = 0,
        [FromQuery] int take = 50,
        CancellationToken ct = default)
    {
        if (skip < 0) skip = 0;
        if (take <= 0) take = 50;
        if (take > 200) take = 200;

        var query = _db.ErrorReports.AsNoTracking();

        if (onlyApproved.HasValue)
            query = query.Where(x => x.Approved == onlyApproved.Value);

        var items = await query
            .OrderByDescending(x => x.CreatedAt)
            .Skip(skip)
            .Take(take)
            .Select(x => new AdminErrorReportListItemDto(
                x.Id,
                x.UserId,
                x.CreatedAt,
                x.FramesCount,
                x.Approved
            ))
            .ToListAsync(ct);

        return Ok(items);
    }

    [HttpGet("{reportId:int}")]
    public async Task<ActionResult<AdminErrorReportDetailsDto>> GetById(
        int reportId,
        CancellationToken ct = default)
    {
        var report = await _db.ErrorReports
            .AsNoTracking()
            .Include(x => x.Video)
            .FirstOrDefaultAsync(x => x.Id == reportId, ct);

        if (report is null)
            return NotFound();

        if (report.Video is null)
            return BadRequest("VideoSample not found for this report.");

        var datasetRoot = GetExtractedDatasetRoot(report.Video.VideoPath);
        var visibleFramesCount = GetVisibleFrameImages(datasetRoot).Count;

        return Ok(new AdminErrorReportDetailsDto(
            report.Id,
            report.UserId,
            report.CreatedAt,
            visibleFramesCount,
            report.Approved,
            report.Comment,
            Path.Combine(report.Video.VideoPath, "extracted").Replace("\\", "/")
        ));
    }

    [HttpGet("{reportId:int}/frames/{frameIndex:int}")]
    public async Task<IActionResult> GetFrameImage(
        int reportId,
        int frameIndex,
        CancellationToken ct = default)
    {
        if (frameIndex <= 0)
            return BadRequest("frameIndex must be >= 1.");

        var frame = await FindVisibleFrameAsync(reportId, frameIndex, ct);
        if (frame.Result is not null)
            return frame.Result;

        var imagePath = frame.Value;
        if (string.IsNullOrWhiteSpace(imagePath))
            return NotFound("Frame image not found.");

        var stream = System.IO.File.OpenRead(imagePath);
        return File(stream, GetImageContentType(imagePath));
    }

    [HttpGet("{reportId:int}/frames/{frameIndex:int}/bbox")]
    public async Task<ActionResult<YoloBboxDto>> GetFrameBbox(
        int reportId,
        int frameIndex,
        CancellationToken ct = default)
    {
        var bboxes = await GetBboxesForVisibleFrameAsync(reportId, frameIndex, ct);

        if (bboxes.Result is not null)
            return bboxes.Result;

        var first = bboxes.Value?.FirstOrDefault();

        if (first is null)
            return NotFound("BBox label not found or empty.");

        return Ok(first);
    }

    [HttpGet("{reportId:int}/frames/{frameIndex:int}/bboxes")]
    public async Task<ActionResult<List<YoloBboxDto>>> GetFrameBboxes(
        int reportId,
        int frameIndex,
        CancellationToken ct = default)
    {
        return await GetBboxesForVisibleFrameAsync(reportId, frameIndex, ct);
    }

    [HttpPut("{reportId:int}/approve")]
    public async Task<IActionResult> SetApproved(
        int reportId,
        [FromBody] ApproveErrorReportRequest req,
        CancellationToken ct = default)
    {
        var report = await _db.ErrorReports.FirstOrDefaultAsync(x => x.Id == reportId, ct);
        if (report is null)
            return NotFound();

        report.Approved = req.Approved;
        await _db.SaveChangesAsync(ct);

        return NoContent();
    }

    [HttpDelete("{reportId:int}")]
    public async Task<IActionResult> DeleteReport(
        int reportId,
        CancellationToken ct = default)
    {
        var report = await _db.ErrorReports
            .Include(x => x.Video)
            .FirstOrDefaultAsync(x => x.Id == reportId, ct);

        if (report is null)
            return NotFound();

        string? datasetRoot = null;
        if (report.Video is not null)
        {
            datasetRoot = Path.Combine(
                _env.ContentRootPath,
                report.Video.VideoPath.Replace("/", Path.DirectorySeparatorChar.ToString())
            );
        }

        _db.ErrorReports.Remove(report);

        if (report.Video is not null)
            _db.VideoSamples.Remove(report.Video);

        await _db.SaveChangesAsync(ct);

        try
        {
            if (!string.IsNullOrWhiteSpace(datasetRoot) && Directory.Exists(datasetRoot))
            {
                Directory.Delete(datasetRoot, recursive: true);
            }
        }
        catch
        {
        }

        return NoContent();
    }

    private async Task<ActionResult<string?>> FindVisibleFrameAsync(int reportId, int frameIndex, CancellationToken ct)
    {
        var report = await _db.ErrorReports
            .AsNoTracking()
            .Include(x => x.Video)
            .FirstOrDefaultAsync(x => x.Id == reportId, ct);

        if (report is null)
            return new NotFoundResult();

        if (report.Video is null)
            return new BadRequestObjectResult("VideoSample not found for this report.");

        var datasetRoot = GetExtractedDatasetRoot(report.Video.VideoPath);
        var images = GetVisibleFrameImages(datasetRoot);

        if (frameIndex > images.Count)
            return new NotFoundObjectResult("Frame image not found or was skipped.");

        return images[frameIndex - 1];
    }

    private async Task<ActionResult<List<YoloBboxDto>>> GetBboxesForVisibleFrameAsync(
        int reportId,
        int frameIndex,
        CancellationToken ct)
    {
        if (frameIndex <= 0)
            return new BadRequestObjectResult("frameIndex must be >= 1.");

        var frame = await FindVisibleFrameAsync(reportId, frameIndex, ct);
        if (frame.Result is not null)
            return frame.Result;

        var imagePath = frame.Value;
        if (string.IsNullOrWhiteSpace(imagePath))
            return new NotFoundObjectResult("Frame image not found.");

        var datasetRoot = Directory.GetParent(Directory.GetParent(imagePath)!.FullName)!.FullName;
        var labelPath = Path.Combine(
            datasetRoot,
            "labels",
            Path.ChangeExtension(Path.GetFileName(imagePath), ".txt")
        );

        if (!System.IO.File.Exists(labelPath))
            return new List<YoloBboxDto>();

        return await ReadYoloBoxesAsync(labelPath, ct);
    }

    private string GetExtractedDatasetRoot(string videoPath) => Path.Combine(
        _env.ContentRootPath,
        videoPath.Replace("/", Path.DirectorySeparatorChar.ToString()),
        "extracted"
    );

    private static List<string> GetVisibleFrameImages(string datasetRoot)
    {
        var imagesDir = Path.Combine(datasetRoot, "images");
        if (!Directory.Exists(imagesDir))
            return new List<string>();

        return Directory.GetFiles(imagesDir, "*.*", SearchOption.TopDirectoryOnly)
            .Where(IsSupportedImage)
            .Where(IsVisibleReportFrame)
            .OrderBy(GetFrameSortKey)
            .ThenBy(x => Path.GetFileName(x), StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static bool IsVisibleReportFrame(string path)
    {
        var fileName = Path.GetFileName(path);
        return !fileName.StartsWith("validation_", StringComparison.OrdinalIgnoreCase);
    }

    private static int GetFrameSortKey(string path)
    {
        var name = Path.GetFileNameWithoutExtension(path);
        const string prefix = "frame_";
        if (name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) &&
            int.TryParse(name[prefix.Length..], out var n))
        {
            return n;
        }

        return int.MaxValue;
    }

    private static bool IsSupportedImage(string path) =>
        path.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
        path.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
        path.EndsWith(".png", StringComparison.OrdinalIgnoreCase);

    private static string GetImageContentType(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext == ".png" ? "image/png" : "image/jpeg";
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
}
