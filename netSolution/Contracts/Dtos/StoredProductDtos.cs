namespace Contracts.Dtos;

public sealed class StoredProductCreateRequest
{
    public int? ProductId { get; init; }
    public string? ProductName { get; init; }
    public DateTime? ManufactureAt { get; init; }
    public DateTime? ExpiryAt { get; init; }
}

public sealed class ProductSummaryDto
{
    public int Id { get; init; }
    public string Name { get; init; } = string.Empty;
    public string? Manufacturer { get; init; }
    public string? Barcode { get; init; }
}

public sealed class StoredProductDto
{
    public int Id { get; init; }
    public int UserId { get; init; }
    public int ProductId { get; init; }
    public DateTime? ManufactureAt { get; init; }
    public DateTime? ExpiryAt { get; init; }
    public DateTime CreatedAt { get; init; }
    public ProductSummaryDto? Product { get; init; }
}
