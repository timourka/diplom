using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Models.Entities;
using PostgreSQLRepository;
using ProductsDateAPI.Services;

namespace ProductsDateAPI.Controllers.Admin;

[ApiController]
[Route("api/admin/training")]
[Authorize]
public class AdminTrainingController : ControllerBase
{
    private readonly AppDbContext _db;
    private readonly TrainingDatasetPackager _packager;
    private readonly TrainingFileStorage _storage;

    public AdminTrainingController(
        AppDbContext db,
        TrainingDatasetPackager packager,
        TrainingFileStorage storage)
    {
        _db = db;
        _packager = packager;
        _storage = storage;
    }

    [HttpPost("start")]
    public async Task<ActionResult<TrainingJobStartResponse>> StartTraining(
        [FromBody] StartTrainingRequest? request,
        CancellationToken ct)
    {
        TrainingDatasetPackage? package = null;
        var jobId = Guid.NewGuid().ToString("N");

        try
        {
            package = await _packager.CreateApprovedReportsZipAsync(ct);
            var datasetZipPath = _storage.SaveDatasetZip(jobId, package.ZipPath);

            var job = new TrainingJob
            {
                JobId = jobId,
                Status = "queued",
                Message = $"Задача создана на сервере. Python-клиент заберёт её при следующем опросе. В датасете {package.ImagesCount} кадров.",
                CreatedAt = DateTime.UtcNow,
                ImagesCount = package.ImagesCount,
                DatasetZipPath = datasetZipPath,
                BaseModel = string.IsNullOrWhiteSpace(request?.BaseModel) ? "yolov8n.pt" : request!.BaseModel,
                Epochs = request?.Epochs ?? 50,
                ImgSize = request?.ImgSize ?? 640,
                Batch = request?.Batch ?? 16,
                Device = string.IsNullOrWhiteSpace(request?.Device) ? "auto" : request!.Device,
                ExportInt8 = request?.ExportInt8 ?? true,
                ExportNms = request?.ExportNms ?? true,
                MobileFormat = string.IsNullOrWhiteSpace(request?.MobileFormat) ? "tflite" : request!.MobileFormat,
                QuantizationFraction = request?.QuantizationFraction ?? 0.3,
            };

            _db.TrainingJobs.Add(job);
            await _db.SaveChangesAsync(ct);

            return Ok(new TrainingJobStartResponse(
                job.JobId,
                job.Status,
                job.ImagesCount,
                job.Message ?? "Задача обучения создана."));
        }
        finally
        {
            if (package is not null && Directory.Exists(package.WorkingDirectory))
            {
                try
                {
                    Directory.Delete(package.WorkingDirectory, recursive: true);
                }
                catch
                {
                    // temp cleanup best effort
                }
            }
        }
    }

    [HttpGet("jobs")]
    public async Task<ActionResult<List<TrainingJobStatusResponse>>> GetJobs(CancellationToken ct)
    {
        var jobs = await _db.TrainingJobs
            .AsNoTracking()
            .OrderByDescending(x => x.CreatedAt)
            .ToListAsync(ct);

        return Ok(jobs.Select(ToJobStatusResponse).ToList());
    }

    [HttpGet("jobs/{jobId}")]
    public async Task<ActionResult<TrainingJobStatusResponse>> GetJob(string jobId, CancellationToken ct)
    {
        var job = await _db.TrainingJobs.AsNoTracking().FirstOrDefaultAsync(x => x.JobId == jobId, ct);
        return job is null ? NotFound() : Ok(ToJobStatusResponse(job));
    }

    [HttpPost("jobs/{jobId}/cancel")]
    public async Task<ActionResult<TrainingJobStatusResponse>> CancelJob(string jobId, CancellationToken ct)
    {
        var job = await _db.TrainingJobs.FirstOrDefaultAsync(x => x.JobId == jobId, ct);
        if (job is null)
            return NotFound();

        if (IsTerminal(job.Status))
            return Ok(ToJobStatusResponse(job));

        job.CancellationRequested = true;
        job.Message = "Администратор запросил остановку задачи. Если задача ещё не началась, она будет отменена сразу; если уже идёт обучение, Python-клиент увидит запрос при следующей синхронизации.";

        if (string.Equals(job.Status, "queued", StringComparison.OrdinalIgnoreCase))
        {
            job.Status = "canceled";
            job.FinishedAt = DateTime.UtcNow;
            job.Message = "Задача отменена администратором до старта обучения.";
        }

        await _db.SaveChangesAsync(ct);
        return Ok(ToJobStatusResponse(job));
    }

    [HttpGet("model-versions")]
    public async Task<ActionResult<List<ModelVersionAdminResponse>>> GetModelVersions(CancellationToken ct)
    {
        var models = await _db.ModelVersions
            .AsNoTracking()
            .Where(x => !x.IsDeleted)
            .OrderByDescending(x => x.IsPublished)
            .ThenByDescending(x => x.TrainedAt)
            .ToListAsync(ct);

        return Ok(models.Select(ToModelVersionResponse).ToList());
    }

    [HttpPost("model-versions/{id:int}/publish")]
    public async Task<ActionResult<ModelVersionAdminResponse>> PublishModelVersion(int id, CancellationToken ct)
    {
        var selected = await _db.ModelVersions.FirstOrDefaultAsync(x => x.Id == id && !x.IsDeleted, ct);
        if (selected is null)
            return NotFound();

        if (string.IsNullOrWhiteSpace(selected.MobileModelPath))
            return BadRequest("У этой версии нет мобильного файла модели.");

        var all = await _db.ModelVersions.Where(x => !x.IsDeleted).ToListAsync(ct);
        foreach (var model in all)
            model.IsPublished = model.Id == selected.Id;

        selected.IsPinned = true;
        await _db.SaveChangesAsync(ct);

        return Ok(ToModelVersionResponse(selected));
    }

    [HttpPost("model-versions/{id:int}/unpublish")]
    public async Task<ActionResult<ModelVersionAdminResponse>> UnpublishModelVersion(int id, CancellationToken ct)
    {
        var selected = await _db.ModelVersions.FirstOrDefaultAsync(x => x.Id == id && !x.IsDeleted, ct);
        if (selected is null)
            return NotFound();

        selected.IsPublished = false;
        await _db.SaveChangesAsync(ct);

        return Ok(ToModelVersionResponse(selected));
    }

    [HttpPost("model-versions/{id:int}/pin")]
    public async Task<ActionResult<ModelVersionAdminResponse>> PinModelVersion(int id, CancellationToken ct)
    {
        var selected = await _db.ModelVersions.FirstOrDefaultAsync(x => x.Id == id && !x.IsDeleted, ct);
        if (selected is null)
            return NotFound();

        selected.IsPinned = true;
        await _db.SaveChangesAsync(ct);
        return Ok(ToModelVersionResponse(selected));
    }

    [HttpPost("model-versions/{id:int}/unpin")]
    public async Task<ActionResult<ModelVersionAdminResponse>> UnpinModelVersion(int id, CancellationToken ct)
    {
        var selected = await _db.ModelVersions.FirstOrDefaultAsync(x => x.Id == id && !x.IsDeleted, ct);
        if (selected is null)
            return NotFound();

        selected.IsPinned = false;
        await _db.SaveChangesAsync(ct);
        return Ok(ToModelVersionResponse(selected));
    }

    [HttpDelete("model-versions/{id:int}")]
    public async Task<IActionResult> DeleteModelVersion(int id, [FromQuery] bool force = false, CancellationToken ct = default)
    {
        var selected = await _db.ModelVersions.FirstOrDefaultAsync(x => x.Id == id && !x.IsDeleted, ct);
        if (selected is null)
            return NotFound();

        if (selected.IsPinned && !force)
            return BadRequest("Версия зафиксирована. Сначала снимите фиксацию или передайте force=true.");

        selected.IsDeleted = true;
        selected.IsPublished = false;
        selected.DeletedAt = DateTime.UtcNow;
        await _db.SaveChangesAsync(ct);

        _storage.TryDeleteFile(selected.MobileModelPath);
        _storage.TryDeleteFile(selected.BestWeightsPath);

        return NoContent();
    }

    private static bool IsTerminal(string? status)
        => string.Equals(status, "completed", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "failed", StringComparison.OrdinalIgnoreCase)
           || string.Equals(status, "canceled", StringComparison.OrdinalIgnoreCase);

    private static TrainingJobStatusResponse ToJobStatusResponse(TrainingJob job)
        => new(
            job.JobId,
            job.Status,
            job.Message,
            job.CreatedAt,
            job.StartedAt,
            job.FinishedAt,
            job.ImagesCount,
            job.BaseModel,
            job.BestWeightsPath,
            job.MobileModelPath,
            job.MobileFormat,
            job.MetricsJson,
            job.CancellationRequested,
            job.ClientId,
            job.AssignedAt,
            job.HeartbeatAt,
            job.DatasetZipPath,
            job.Epochs,
            job.ImgSize,
            job.Batch,
            job.Device,
            job.ExportInt8,
            job.ExportNms,
            job.QuantizationFraction,
            job.MobileModelFileName,
            job.MobileModelContentType);

    private static ModelVersionAdminResponse ToModelVersionResponse(ModelVersion model)
        => new(
            model.Id,
            model.ExternalJobId,
            model.TrainedAt,
            model.MetricsJson,
            model.BaseModel,
            model.BestWeightsPath,
            model.MobileModelPath,
            model.MobileModelFileName,
            model.MobileModelContentType,
            model.MobileFormat,
            model.IsPublished,
            model.IsPinned,
            model.IsDeleted,
            model.DeletedAt);
}
