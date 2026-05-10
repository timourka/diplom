using Contracts.Dtos;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PostgreSQLRepository;
using ProductsDateAPI.Services;

namespace ProductsDateAPI.Controllers;

[ApiController]
[Route("api/mobile-models")]
public class MobileModelsController : ControllerBase
{
    private readonly AppDbContext _db;
    private readonly TrainingFileStorage _storage;

    public MobileModelsController(AppDbContext db, TrainingFileStorage storage)
    {
        _db = db;
        _storage = storage;
    }

    [HttpGet("latest")]
    public async Task<ActionResult<LatestMobileModelResponse>> GetLatest(CancellationToken ct)
    {
        var model = await _db.ModelVersions
            .AsNoTracking()
            .Where(x => x.IsPublished && !x.IsDeleted && x.MobileModelPath != null)
            .OrderByDescending(x => x.TrainedAt)
            .FirstOrDefaultAsync(ct);

        if (model is null)
            return NotFound();

        return Ok(new LatestMobileModelResponse(
            model.Id,
            model.TrainedAt,
            model.MobileFormat,
            model.MetricsJson,
            model.MobileModelFileName,
            model.IsPinned
        ));
    }

    [HttpGet("latest/download")]
    public async Task<IActionResult> DownloadLatest(CancellationToken ct)
    {
        var model = await _db.ModelVersions
            .AsNoTracking()
            .Where(x => x.IsPublished && !x.IsDeleted && x.MobileModelPath != null)
            .OrderByDescending(x => x.TrainedAt)
            .FirstOrDefaultAsync(ct);

        if (model is null || string.IsNullOrWhiteSpace(model.MobileModelPath))
            return NotFound();

        var absolute = _storage.ToAbsolutePath(model.MobileModelPath);
        if (!System.IO.File.Exists(absolute))
            return NotFound("Published model file is missing on backend storage.");

        var fileName = string.IsNullOrWhiteSpace(model.MobileModelFileName)
            ? $"latest_model.{model.MobileFormat ?? "bin"}"
            : model.MobileModelFileName;

        return File(
            System.IO.File.OpenRead(absolute),
            string.IsNullOrWhiteSpace(model.MobileModelContentType) ? "application/octet-stream" : model.MobileModelContentType,
            fileName,
            enableRangeProcessing: true);
    }
}
