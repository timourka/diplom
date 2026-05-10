using System.ComponentModel.DataAnnotations;

namespace Models.Entities;

public class ModelVersion : BaseEntity
{
    [Required]
    public DateTime TrainedAt { get; set; } = DateTime.UtcNow;

    /// <summary>Метрики в JSON (например: mAP/precision/recall)</summary>
    public string? MetricsJson { get; set; }

    [MaxLength(128)]
    public string? ExternalJobId { get; set; }

    [MaxLength(256)]
    public string? BaseModel { get; set; }

    [MaxLength(2048)]
    public string? BestWeightsPath { get; set; }

    [MaxLength(2048)]
    public string? MobileModelPath { get; set; }

    [MaxLength(32)]
    public string? MobileFormat { get; set; }

    [MaxLength(512)]
    public string? MobileModelFileName { get; set; }

    [MaxLength(128)]
    public string? MobileModelContentType { get; set; }

    public bool IsPublished { get; set; }

    public bool IsPinned { get; set; }

    public bool IsDeleted { get; set; }

    public DateTime? DeletedAt { get; set; }

    public ICollection<ErrorReport> ErrorReports { get; set; } = new List<ErrorReport>();
}
