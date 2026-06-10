namespace ProductsDateAPI.Models;

public class UploadDatasetForm
{
    public IFormFile datasetZip { get; set; } = default!;
    public string? comment { get; set; }
    public string? validationToken { get; set; }
    public string? validationFrameName { get; set; }
}
