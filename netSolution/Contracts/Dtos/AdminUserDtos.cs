namespace Contracts.Dtos;

public record AdminUserListItem(int Id, string Email, bool IsBlocked, DateTime? CreatedAt);
public record SetBlockedRequest(bool IsBlocked);
