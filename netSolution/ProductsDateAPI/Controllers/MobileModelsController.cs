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
    private readonly TrainingServiceClient _trainingClient;

    public MobileModelsController(AppDbContext db, TrainingServiceClient trainingClient)
    {
        _db = db;
        _trainingClient = trainingClient;
    }

    [HttpGet("latest")]
    public async Task<ActionResult<LatestMobileModelResponse>> GetLatest(CancellationToken ct)
    {
        var model = await _db.ModelVersions
            .AsNoTracking()
            .Where(x => x.MobileModelPath != null && x.ExternalJobId != null)
            .OrderByDescending(x => x.TrainedAt)
            .FirstOrDefaultAsync(ct);

        if (model is null)
            return NotFound();

        return Ok(new LatestMobileModelResponse(
            model.Id,
            model.TrainedAt,
            model.MobileFormat,
            model.MetricsJson
        ));
    }

    [HttpGet("latest/download")]
    public async Task<IActionResult> DownloadLatest(CancellationToken ct)
    {
        var model = await _db.ModelVersions
            .AsNoTracking()
            .Where(x => x.MobileModelPath != null && x.ExternalJobId != null)
            .OrderByDescending(x => x.TrainedAt)
            .FirstOrDefaultAsync(ct);

        if (model is null || string.IsNullOrWhiteSpace(model.ExternalJobId))
            return NotFound();

        var artifact = await _trainingClient.DownloadArtifactAsync(model.ExternalJobId, "mobile", ct);
        var fileName = string.IsNullOrWhiteSpace(artifact.FileName)
            ? $"latest_model.{model.MobileFormat ?? "bin"}"
            : artifact.FileName;

        return File(
            artifact.Bytes,
            artifact.ContentType ?? "application/octet-stream",
            fileName,
            enableRangeProcessing: false);
    }
}
