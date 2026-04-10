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
    private readonly TrainingServiceClient _trainingClient;

    public AdminTrainingController(
        AppDbContext db,
        TrainingDatasetPackager packager,
        TrainingServiceClient trainingClient)
    {
        _db = db;
        _packager = packager;
        _trainingClient = trainingClient;
    }

    [HttpPost("start")]
    public async Task<ActionResult<TrainingJobStartResponse>> StartTraining(
        [FromBody] StartTrainingRequest? request,
        CancellationToken ct)
    {
        TrainingDatasetPackage? package = null;

        try
        {
            package = await _packager.CreateApprovedReportsZipAsync(ct);
            var remote = await _trainingClient.StartTrainingAsync(package.ZipPath, request, ct);

            return Ok(remote with
            {
                ImagesCount = package.ImagesCount,
                Message = $"{remote.Message}. В отправленном датасете {package.ImagesCount} кадров."
            });
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
        var jobs = await _trainingClient.GetJobsAsync(ct);
        await SyncCompletedJobsAsync(jobs, ct);
        return Ok(jobs.OrderByDescending(x => x.CreatedAt).ToList());
    }

    [HttpGet("jobs/{jobId}")]
    public async Task<ActionResult<TrainingJobStatusResponse>> GetJob(string jobId, CancellationToken ct)
    {
        var remote = await _trainingClient.GetJobAsync(jobId, ct);
        if (remote is null)
            return NotFound();

        if (string.Equals(remote.Status, "completed", StringComparison.OrdinalIgnoreCase))
        {
            await SyncModelVersionAsync(remote, ct);
        }

        return Ok(remote);
    }

    [HttpPost("jobs/{jobId}/cancel")]
    public async Task<ActionResult<TrainingJobStatusResponse>> CancelJob(string jobId, CancellationToken ct)
    {
        var remote = await _trainingClient.CancelJobAsync(jobId, ct);
        return Ok(remote);
    }

    private async Task SyncCompletedJobsAsync(IEnumerable<TrainingJobStatusResponse> jobs, CancellationToken ct)
    {
        foreach (var job in jobs)
        {
            if (string.Equals(job.Status, "completed", StringComparison.OrdinalIgnoreCase))
                await SyncModelVersionAsync(job, ct);
        }
    }

    private async Task SyncModelVersionAsync(TrainingJobStatusResponse remote, CancellationToken ct)
    {
        var exists = await _db.ModelVersions.AnyAsync(x => x.ExternalJobId == remote.JobId, ct);
        if (exists)
            return;

        var entity = new ModelVersion
        {
            ExternalJobId = remote.JobId,
            TrainedAt = remote.FinishedAt ?? DateTime.UtcNow,
            MetricsJson = remote.MetricsJson,
            BaseModel = remote.BaseModel,
            BestWeightsPath = remote.BestWeightsPath,
            MobileModelPath = "artifacts/mobile.tflite",
            MobileFormat = remote.MobileFormat
        };

        _db.ModelVersions.Add(entity);
        await _db.SaveChangesAsync(ct);
    }
}
