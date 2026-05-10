using System.Text.Json;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using ProductsDateAPI.Models;
using ProductsDateAPI.Services;

namespace ProductsDateAPI.Controllers.Admin;

[ApiController]
[Route("api/admin/backup")]
[Authorize]
public class AdminBackupController : ControllerBase
{
    private readonly BackupService _backupService;

    public AdminBackupController(BackupService backupService)
    {
        _backupService = backupService;
    }

    [HttpGet("export")]
    [Produces("application/zip")]
    public async Task<IActionResult> Export(CancellationToken ct)
    {
        var zipPath = await _backupService.CreateBackupZipAsync(ct);
        var fileName = $"productsdate_backup_{DateTime.UtcNow:yyyyMMdd_HHmmss}.zip";
        var stream = new FileStream(
            zipPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 1024 * 128,
            options: FileOptions.Asynchronous | FileOptions.DeleteOnClose);

        return File(stream, "application/zip", fileName);
    }

    [HttpPost("import")]
    [Consumes("multipart/form-data")]
    [RequestSizeLimit(2_000_000_000)]
    public async Task<ActionResult<BackupImportResult>> Import([FromForm] BackupImportForm form, CancellationToken ct)
    {
        if (form.BackupZip is null || form.BackupZip.Length == 0)
            return BadRequest("BackupZip is required.");

        try
        {
            var result = await _backupService.ImportBackupZipAsync(form.BackupZip, form.ReplaceExisting, ct);
            return Ok(result);
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ex.Message);
        }
        catch (InvalidDataException ex)
        {
            return BadRequest($"Не удалось прочитать ZIP-файл: {ex.Message}");
        }
        catch (JsonException ex)
        {
            return BadRequest($"Не удалось прочитать JSON внутри backup-архива: {ex.Message}");
        }
    }
}
