namespace Contracts.Dtos;

public record UserProfileResponse(int Id, string Email, bool IsBlocked, string? SettingsJson);
