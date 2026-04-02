namespace ProductsDateAPI.Models;

public class TrainingServiceOptions
{
    public string BaseUrl { get; set; } = "http://127.0.0.1:8001/";
    public string? ApiKey { get; set; }
    public string TempRoot { get; set; } = "temp/training";
    public string? SeedDatasetPath { get; set; }
}
