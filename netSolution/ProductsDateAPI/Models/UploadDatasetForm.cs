namespace ProductsDateAPI.Models;

public class UploadDatasetForm
{
    public IFormFile datasetZip { get; set; } = default!;
    public string? comment { get; set; }
}