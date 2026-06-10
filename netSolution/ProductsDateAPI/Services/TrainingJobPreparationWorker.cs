using Microsoft.EntityFrameworkCore;
using PostgreSQLRepository;

namespace ProductsDateAPI.Services;

public sealed class TrainingJobPreparationWorker : BackgroundService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ITrainingJobPreparationQueue _queue;
    private readonly ILogger<TrainingJobPreparationWorker> _logger;

    public TrainingJobPreparationWorker(
        IServiceScopeFactory scopeFactory,
        ITrainingJobPreparationQueue queue,
        ILogger<TrainingJobPreparationWorker> logger)
    {
        _scopeFactory = scopeFactory;
        _queue = queue;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var jobId = await _queue.DequeueAsync(TimeSpan.FromSeconds(3), stoppingToken)
                            ?? await FindNextPreparingJobIdAsync(stoppingToken);

                if (string.IsNullOrWhiteSpace(jobId))
                    continue;

                await PrepareJobAsync(jobId, stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                return;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unexpected training job preparation worker error.");
                await Task.Delay(TimeSpan.FromSeconds(3), stoppingToken);
            }
        }
    }

    private async Task<string?> FindNextPreparingJobIdAsync(CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        return await db.TrainingJobs
            .AsNoTracking()
            .Where(x => x.Status == "preparing" && !x.CancellationRequested)
            .OrderBy(x => x.CreatedAt)
            .Select(x => x.JobId)
            .FirstOrDefaultAsync(ct);
    }

    private async Task PrepareJobAsync(string jobId, CancellationToken ct)
    {
        TrainingDatasetPackage? package = null;

        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
        var packager = scope.ServiceProvider.GetRequiredService<TrainingDatasetPackager>();
        var storage = scope.ServiceProvider.GetRequiredService<TrainingFileStorage>();

        var job = await db.TrainingJobs.FirstOrDefaultAsync(x => x.JobId == jobId, ct);
        if (job is null)
            return;

        if (!string.Equals(job.Status, "preparing", StringComparison.OrdinalIgnoreCase))
            return;

        if (job.CancellationRequested)
        {
            job.Status = "canceled";
            job.FinishedAt = DateTime.UtcNow;
            job.Message = "Задача отменена до подготовки датасета.";
            await db.SaveChangesAsync(ct);
            return;
        }

        try
        {
            job.Message = "Подготовка датасета.";
            await db.SaveChangesAsync(ct);

            package = await packager.CreateApprovedReportsZipAsync(ct);
            var datasetZipPath = storage.SaveDatasetZip(job.JobId, package.ZipPath);

            await db.Entry(job).ReloadAsync(ct);
            if (job.CancellationRequested)
            {
                job.Status = "canceled";
                job.FinishedAt = DateTime.UtcNow;
                job.Message = "Задача отменена до старта обучения.";
            }
            else
            {
                job.Status = "queued";
                job.ImagesCount = package.ImagesCount;
                job.DatasetZipPath = datasetZipPath;
                job.Message = $"Датасет готов. Кадров: {package.ImagesCount}.";
            }

            await db.SaveChangesAsync(ct);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to prepare training dataset for job {JobId}.", jobId);

            try
            {
                await db.Entry(job).ReloadAsync(CancellationToken.None);
                if (string.Equals(job.Status, "preparing", StringComparison.OrdinalIgnoreCase))
                {
                    job.Status = "failed";
                    job.FinishedAt = DateTime.UtcNow;
                    job.Message = "Не удалось подготовить датасет: " + ex.Message;
                    await db.SaveChangesAsync(CancellationToken.None);
                }
            }
            catch (Exception saveEx)
            {
                _logger.LogError(saveEx, "Failed to save training dataset preparation error for job {JobId}.", jobId);
            }
        }
        finally
        {
            if (package is not null && Directory.Exists(package.WorkingDirectory))
            {
                try
                {
                    Directory.Delete(package.WorkingDirectory, recursive: true);
                }
                catch (Exception cleanupEx)
                {
                    _logger.LogWarning(cleanupEx, "Failed to cleanup training temp directory {Directory}.", package.WorkingDirectory);
                }
            }
        }
    }
}
