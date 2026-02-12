using System.ComponentModel.DataAnnotations;

namespace Models.Entities;

public class Product : BaseEntity
{
    [Required, MaxLength(256)]
    public string Name { get; set; } = string.Empty;

    [MaxLength(256)]
    public string? Manufacturer { get; set; }

    [MaxLength(64)]
    public string? Barcode { get; set; }

    public ICollection<StoredProduct> StoredProducts { get; set; } = new List<StoredProduct>();
}
