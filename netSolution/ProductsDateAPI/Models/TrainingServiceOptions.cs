namespace ProductsDateAPI.Models;

public class TrainingServiceOptions
{
    public string? ApiKey { get; set; }
    public string TempRoot { get; set; } = "temp/training";
    public string StorageRoot { get; set; } = "storage/training";
    public string? SeedDatasetPath { get; set; }
}
