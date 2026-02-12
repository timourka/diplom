using System.ComponentModel.DataAnnotations;

namespace Models.Entities;

public class User : BaseEntity
{
    [Required, EmailAddress, MaxLength(256)]
    public string Email { get; set; } = string.Empty;

    [Required, MaxLength(256)]
    public string PasswordHash { get; set; } = string.Empty;

    public bool IsBlocked { get; set; } = false;

    /// <summary>Расширяемые настройки в JSON (может быть null/пусто)</summary>
    public string? SettingsJson { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public ICollection<StoredProduct> StoredProducts { get; set; } = new List<StoredProduct>();
    public ICollection<ErrorReport> ErrorReports { get; set; } = new List<ErrorReport>();
}
