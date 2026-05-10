using System.ComponentModel.DataAnnotations;

namespace Models.Entities;

public class TrainingJob : BaseEntity
{
    [Required]
    [MaxLength(128)]
    public string JobId { get; set; } = Guid.NewGuid().ToString("N");

    [Required]
    [MaxLength(32)]
    public string Status { get; set; } = "queued";

    public string? Message { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? StartedAt { get; set; }
    public DateTime? FinishedAt { get; set; }
    public DateTime? AssignedAt { get; set; }
    public DateTime? HeartbeatAt { get; set; }

    public int ImagesCount { get; set; }

    [MaxLength(256)]
    public string? ClientId { get; set; }

    [MaxLength(2048)]
    public string DatasetZipPath { get; set; } = string.Empty;

    [MaxLength(256)]
    public string? BaseModel { get; set; }

    public int Epochs { get; set; } = 50;
    public int ImgSize { get; set; } = 640;
    public int Batch { get; set; } = 16;

    [MaxLength(64)]
    public string? Device { get; set; } = "auto";

    public bool ExportInt8 { get; set; } = true;
    public bool ExportNms { get; set; } = true;

    [MaxLength(32)]
    public string? MobileFormat { get; set; } = "tflite";

    public double QuantizationFraction { get; set; } = 0.3;

    [MaxLength(2048)]
    public string? BestWeightsPath { get; set; }

    [MaxLength(2048)]
    public string? MobileModelPath { get; set; }

    [MaxLength(512)]
    public string? MobileModelFileName { get; set; }

    [MaxLength(128)]
    public string? MobileModelContentType { get; set; }

    public string? MetricsJson { get; set; }

    public bool CancellationRequested { get; set; }
}
