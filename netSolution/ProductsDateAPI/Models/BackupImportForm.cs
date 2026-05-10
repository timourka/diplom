using Microsoft.AspNetCore.Http;

namespace ProductsDateAPI.Models;

public class BackupImportForm
{
    public IFormFile? BackupZip { get; set; }
    public bool ReplaceExisting { get; set; } = false;
}
