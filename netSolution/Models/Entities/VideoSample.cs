using System.ComponentModel.DataAnnotations;

namespace Models.Entities;

public class VideoSample : BaseEntity
{
    [Required, MaxLength(1024)]
    public string VideoPath { get; set; } = string.Empty;

    /// <summary>camera/file/test и т.п.</summary>
    [MaxLength(64)]
    public string? Source { get; set; }

    public ICollection<ErrorReport> ErrorReports { get; set; } = new List<ErrorReport>();
}
