using System.ComponentModel.DataAnnotations;

namespace Models.Entities;

public class ErrorReport : BaseEntity
{
    [Required]
    public int UserId { get; set; }
    public User? User { get; set; }

    public int? VideoId { get; set; }
    public VideoSample? Video { get; set; }

    public int? ModelVersionId { get; set; }
    public ModelVersion? ModelVersion { get; set; }

    [MaxLength(2000)]
    public string? Comment { get; set; }

    [Required]
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public bool Resolved { get; set; } = false;

    public bool Approved { get; set; } = false;
    public int FramesCount { get; set; } = 0;
}
