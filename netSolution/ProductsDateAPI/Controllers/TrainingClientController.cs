using Contracts.Dtos;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Models.Entities;
using PostgreSQLRepository;
using ProductsDateAPI.Models;
using ProductsDateAPI.Services;

namespace ProductsDateAPI.Controllers;

[ApiController]
[Route("api/training-client")]
public class TrainingClientController : ControllerBase
{
    private readonly AppDbContext _db;
    private readonly TrainingFileStorage _storage;
    private readonly TrainingServiceOptions _options;

    public TrainingClientController(
        AppDbContext db,
        TrainingFileStorage storage,
        IOptions<TrainingServiceOptions> options)
    {
        _db = db;
        _storage = storage;
        _options = options.Value;
    }

    [HttpGet("jobs/next")]
    public async Task<IActionResult> GetNextJob(
        [FromQuery] string? clientId,
        [FromHeader(Name = "X-Training-Client-Key")] string? apiKey,
        CancellationToken ct)
    {
        if (!IsAuthorized(apiKey))
            return Unauthorized();

        var job = await _db.TrainingJobs
            .Where(x => x.Status == "queued" && !x.CancellationRequested)
            .OrderBy(x => x.CreatedAt)
            .FirstOrDefaultAsync(ct);

        if (job is null)
            return NoContent();

        job.Status = "running";
        job.StartedAt ??= DateTime.UtcNow;
        job.AssignedAt = DateTime.UtcNow;
        job.HeartbeatAt = DateTime.UtcNow;
        job.ClientId = string.IsNullOrWhiteSpace(clientId) ? Environment.MachineName : clientId;
        job.Message = $"Задача забрана Python-клиентом {job.ClientId}.";
        await _db.SaveChangesAsync(ct);

        return Ok(new TrainingClientJobResponse(
            job.JobId,
            job.ImagesCount,
            job.BaseModel,
            job.Epochs,
            job.ImgSize,
            job.Batch,
            job.Device,
            job.ExportInt8,
            job.ExportNms,
            job.MobileFormat,
            job.QuantizationFraction,
            job.CancellationRequested));
    }

    [HttpGet("jobs/{jobId}")]
    public async Task<IActionResult> GetJob(
        string jobId,
        [FromHeader(Name = "X-Training-Client-Key")] string? apiKey,
        CancellationToken ct)
    {
        if (!IsAuthorized(apiKey))
            return Unauthorized();

        var job = await _db.TrainingJobs.AsNoTracking().FirstOrDefaultAsync(x => x.JobId == jobId, ct);
        if (job is null)
            return NotFound();

        return Ok(new TrainingClientStatusResponse(job.JobId, job.Status, job.CancellationRequested, job.Message));
    }

    [HttpGet("jobs/{jobId}/dataset")]
    public async Task<IActionResult> DownloadDataset(
        string jobId,
        [FromHeader(Name = "X-Training-Client-Key")] string? apiKey,
        CancellationToken ct)
    {
        if (!IsAuthorized(apiKey))
            return Unauthorized();

        var job = await _db.TrainingJobs.AsNoTracking().FirstOrDefaultAsync(x => x.JobId == jobId, ct);
        if (job is null)
            return NotFound();

        if (string.IsNullOrWhiteSpace(job.DatasetZipPath))
            return NotFound("Dataset file path is empty.");

        var absolute = _storage.ToAbsolutePath(job.DatasetZipPath);
        if (!System.IO.File.Exists(absolute))
            return NotFound("Dataset file is missing on backend storage.");

        return File(System.IO.File.OpenRead(absolute), "application/zip", $"dataset_{job.JobId}.zip");
    }

    [HttpPost("jobs/{jobId}/status")]
    public async Task<IActionResult> UpdateStatus(
        string jobId,
        [FromBody] TrainingClientStatusRequest request,
        [FromHeader(Name = "X-Training-Client-Key")] string? apiKey,
        CancellationToken ct)
    {
        if (!IsAuthorized(apiKey))
            return Unauthorized();

        var job = await _db.TrainingJobs.FirstOrDefaultAsync(x => x.JobId == jobId, ct);
        if (job is null)
            return NotFound();

        if (IsTerminal(job.Status))
        {
            return Ok(new TrainingClientStatusResponse(job.JobId, job.Status, job.CancellationRequested, job.Message));
        }

        job.HeartbeatAt = DateTime.UtcNow;

        if (job.CancellationRequested)
        {
            job.Status = "canceled";
            job.FinishedAt = DateTime.UtcNow;
            job.Message = "Задача остановлена администратором.";

            await _db.SaveChangesAsync(ct);
            return Ok(new TrainingClientStatusResponse(job.JobId, job.Status, job.CancellationRequested, job.Message));
        }

        if (!string.IsNullOrWhiteSpace(request.Status))
        {
            var requestedStatus = request.Status.Trim().ToLowerInvariant();
            if (requestedStatus is "running" or "failed" or "canceled")
            {
                job.Status = requestedStatus;
                if (requestedStatus is "failed" or "canceled")
                    job.FinishedAt = DateTime.UtcNow;
            }
        }

        if (!string.IsNullOrWhiteSpace(request.Message))
            job.Message = request.Message;

        if (!string.IsNullOrWhiteSpace(request.MetricsJson))
            job.MetricsJson = request.MetricsJson;

        await _db.SaveChangesAsync(ct);
        return Ok(new TrainingClientStatusResponse(job.JobId, job.Status, job.CancellationRequested, job.Message));
    }

    [HttpPost("jobs/{jobId}/artifacts")]
    [RequestSizeLimit(2_147_483_648L)]
    [RequestFormLimits(MultipartBodyLengthLimit = 2_147_483_648L)]
    public async Task<IActionResult> UploadArtifacts(
        string jobId,
        [FromForm] TrainingArtifactUploadForm form,
        [FromHeader(Name = "X-Training-Client-Key")] string? apiKey,
        CancellationToken ct)
    {
        if (!IsAuthorized(apiKey))
            return Unauthorized();

        if (form.MobileModel is null || form.MobileModel.Length == 0)
            return BadRequest("mobileModel is required.");

        var job = await _db.TrainingJobs.FirstOrDefaultAsync(x => x.JobId == jobId, ct);
        if (job is null)
            return NotFound();

        if (job.CancellationRequested || IsTerminal(job.Status))
            return Conflict("Training job is canceled or already finished.");

        var bestPath = form.BestWeights is not null && form.BestWeights.Length > 0
            ? await _storage.SaveArtifactAsync(jobId, form.BestWeights, form.BestWeights.FileName, ct)
            : null;
        var mobilePath = await _storage.SaveArtifactAsync(jobId, form.MobileModel, form.MobileModel.FileName, ct);
        var format = string.IsNullOrWhiteSpace(form.MobileFormat) ? job.MobileFormat : form.MobileFormat;

        job.Status = "completed";
        job.FinishedAt = DateTime.UtcNow;
        job.Message = "Python-клиент завершил обучение и загрузил артефакты на бэк.";
        job.BestWeightsPath = bestPath;
        job.MobileModelPath = mobilePath;
        job.MobileModelFileName = form.MobileModel.FileName;
        job.MobileModelContentType = string.IsNullOrWhiteSpace(form.MobileModel.ContentType) ? "application/octet-stream" : form.MobileModel.ContentType;
        job.MobileFormat = format;
        job.MetricsJson = form.MetricsJson;

        var model = await _db.ModelVersions.FirstOrDefaultAsync(x => x.ExternalJobId == job.JobId, ct);
        if (model is null)
        {
            model = new ModelVersion
            {
                ExternalJobId = job.JobId,
                IsPublished = false,
                IsPinned = false,
            };
            _db.ModelVersions.Add(model);
        }

        model.TrainedAt = job.FinishedAt ?? DateTime.UtcNow;
        model.MetricsJson = form.MetricsJson;
        model.BaseModel = job.BaseModel;
        model.BestWeightsPath = bestPath;
        model.MobileModelPath = mobilePath;
        model.MobileModelFileName = form.MobileModel.FileName;
        model.MobileModelContentType = job.MobileModelContentType;
        model.MobileFormat = format;
        model.IsDeleted = false;
        model.DeletedAt = null;

        await _db.SaveChangesAsync(ct);
        return Ok(new TrainingClientStatusResponse(job.JobId, job.Status, job.CancellationRequested, job.Message));
    }

    private bool IsAuthorized(string? apiKey)
        => string.IsNullOrWhiteSpace(_options.ApiKey) || apiKey == _options.ApiKey;

    private static bool IsTerminal(string? status)
        => string.Equals(status, "completed", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "failed", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "canceled", StringComparison.OrdinalIgnoreCase);
}
