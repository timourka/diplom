using Microsoft.AspNetCore.Http;

namespace ProductsDateAPI.Models;

public class TrainingArtifactUploadForm
{
    public IFormFile? BestWeights { get; set; }
    public IFormFile? MobileModel { get; set; }
    public string? MetricsJson { get; set; }
    public string? MobileFormat { get; set; }
}
