using System.Security.Claims;

namespace ProductsDateAPI.Helpers;

public static class ClaimsExtensions
{
    public static int GetUserIdOrThrow(this ClaimsPrincipal user)
    {
        var sub = user.FindFirst("sub")?.Value ?? user.FindFirst(ClaimTypes.NameIdentifier)?.Value;
        if (string.IsNullOrWhiteSpace(sub) || !int.TryParse(sub, out var id))
            throw new InvalidOperationException("Invalid JWT: missing sub");
        return id;
    }
}
