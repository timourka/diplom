namespace Contracts.Dtos;

public record ProductCreateRequest(string Name, string? Manufacturer, string? Barcode);
public record ProductUpdateRequest(string Name, string? Manufacturer, string? Barcode);
