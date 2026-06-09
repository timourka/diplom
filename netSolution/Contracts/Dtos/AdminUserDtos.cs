namespace Contracts.Dtos;

public record AdminUserListItem(int Id, string Login, bool IsBlocked, bool IsAdmin, DateTime? CreatedAt);

public record AdminUserDetailsDto(
    int Id,
    string Login,
    bool IsBlocked,
    bool IsAdmin,
    DateTime? CreatedAt,
    string? SettingsJson,
    int StoredProductsCount,
    int ErrorReportsCount,
    int ApprovedReportsCount);

public record AdminUserUpdateRequest(string Login, bool IsBlocked, bool IsAdmin, string? SettingsJson);

public record SetBlockedRequest(bool IsBlocked);
