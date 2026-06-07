namespace Contracts.Dtos;

public record UserProfileResponse(int Id, string Email, bool IsBlocked, bool IsAdmin, string? SettingsJson);
