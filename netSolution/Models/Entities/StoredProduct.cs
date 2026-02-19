using System.ComponentModel.DataAnnotations;

namespace Models.Entities;

public class StoredProduct : BaseEntity
{
    [Required]
    public int UserId { get; set; }
    public User? User { get; set; }

    [Required]
    public int ProductId { get; set; }
    public Product? Product { get; set; }

    /// <summary>Дата изготовления (если известна)</summary>
    public DateTime? ManufactureAt { get; set; }

    /// <summary>Срок годности (дата истечения)</summary>
    public DateTime? ExpiryAt { get; set; }

    [Required]
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
