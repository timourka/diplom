namespace Contracts.Dtos;

public record AdminUserListItem(int Id, string Email, bool IsBlocked, bool IsAdmin, DateTime? CreatedAt);
public record SetBlockedRequest(bool IsBlocked);
