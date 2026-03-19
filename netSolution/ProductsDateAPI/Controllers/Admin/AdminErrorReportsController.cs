using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PostgreSQLRepository;
using System.Globalization;

namespace ProductsDateAPI.Controllers.Admin;

[ApiController]
[Route("api/admin/error-reports")]
[Authorize]
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

        return Ok(new AdminErrorReportDetailsDto(
            report.Id,
            report.UserId,
            report.CreatedAt,
            report.FramesCount,
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

        var report = await _db.ErrorReports
            .AsNoTracking()
            .Include(x => x.Video)
            .FirstOrDefaultAsync(x => x.Id == reportId, ct);

        if (report is null)
            return NotFound();

        if (report.Video is null)
            return BadRequest("VideoSample not found for this report.");

        var datasetRoot = Path.Combine(
            _env.ContentRootPath,
            report.Video.VideoPath.Replace("/", Path.DirectorySeparatorChar.ToString()),
            "extracted"
        );

        var imagePath = Path.Combine(datasetRoot, "images", $"frame_{frameIndex:D5}.jpg");

        if (!System.IO.File.Exists(imagePath))
            return NotFound("Frame image not found.");

        var stream = System.IO.File.OpenRead(imagePath);
        return File(stream, "image/jpeg");
    }

    [HttpGet("{reportId:int}/frames/{frameIndex:int}/bbox")]
    public async Task<ActionResult<YoloBboxDto>> GetFrameBbox(
        int reportId,
        int frameIndex,
        CancellationToken ct = default)
    {
        if (frameIndex <= 0)
            return BadRequest("frameIndex must be >= 1.");

        var report = await _db.ErrorReports
            .AsNoTracking()
            .Include(x => x.Video)
            .FirstOrDefaultAsync(x => x.Id == reportId, ct);

        if (report is null)
            return NotFound();

        if (report.Video is null)
            return BadRequest("VideoSample not found for this report.");

        var datasetRoot = Path.Combine(
            _env.ContentRootPath,
            report.Video.VideoPath.Replace("/", Path.DirectorySeparatorChar.ToString()),
            "extracted"
        );

        var labelPath = Path.Combine(datasetRoot, "labels", $"frame_{frameIndex:D5}.txt");

        if (!System.IO.File.Exists(labelPath))
            return NotFound("BBox label not found.");

        var line = await System.IO.File.ReadAllTextAsync(labelPath, ct);
        if (string.IsNullOrWhiteSpace(line))
            return BadRequest("BBox label file is empty.");

        var parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 5)
            return BadRequest("Invalid YOLO label format.");

        if (!int.TryParse(parts[0], out var classId))
            return BadRequest("Invalid classId.");

        if (!double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var xc))
            return BadRequest("Invalid xc.");
        if (!double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var yc))
            return BadRequest("Invalid yc.");
        if (!double.TryParse(parts[3], NumberStyles.Float, CultureInfo.InvariantCulture, out var w))
            return BadRequest("Invalid w.");
        if (!double.TryParse(parts[4], NumberStyles.Float, CultureInfo.InvariantCulture, out var h))
            return BadRequest("Invalid h.");

        return Ok(new YoloBboxDto(classId, xc, yc, w, h));
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
}