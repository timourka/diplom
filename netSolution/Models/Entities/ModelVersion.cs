using System.ComponentModel.DataAnnotations;

namespace Models.Entities;

public class ModelVersion : BaseEntity
{
    [Required]
    public DateTime TrainedAt { get; set; } = DateTime.UtcNow;

    /// <summary>Метрики в JSON (например: mAP/precision/recall)</summary>
    public string? MetricsJson { get; set; }

    public ICollection<ErrorReport> ErrorReports { get; set; } = new List<ErrorReport>();
}
