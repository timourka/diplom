namespace Contracts.Dtos;

public record StoredProductCreateRequest(int ProductId, DateTime? ManufactureAt, DateTime? ExpiryAt);
