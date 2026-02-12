using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PostgreSQLRepository;
using ProductsDateAPI.Helpers;

namespace ProductsDateAPI.Controllers;

[ApiController]
[Route("api/users")]
public class UsersController : ControllerBase
{
    private readonly AppDbContext _db;
    public UsersController(AppDbContext db) => _db = db;

    [Authorize]
    [HttpGet("me")]
    public async Task<ActionResult<UserProfileResponse>> Me(CancellationToken ct)
    {
        var userId = User.GetUserIdOrThrow();
        var u = await _db.Users.AsNoTracking().FirstOrDefaultAsync(x => x.Id == userId, ct);
        if (u is null) return NotFound();

        return Ok(new UserProfileResponse(u.Id, u.Email, u.IsBlocked, u.SettingsJson));
    }

    // опционально: обновление своих настроек (JSON)
    [Authorize]
    [HttpPut("me/settings")]
    public async Task<ActionResult> UpdateSettings([FromBody] string? settingsJson, CancellationToken ct)
    {
        var userId = User.GetUserIdOrThrow();
        var u = await _db.Users.FirstOrDefaultAsync(x => x.Id == userId, ct);
        if (u is null) return NotFound();

        u.SettingsJson = settingsJson;
        await _db.SaveChangesAsync(ct);
        return NoContent();
    }

    [Authorize]
    [HttpDelete("me")]
    public async Task<ActionResult> DeleteMe(CancellationToken ct)
    {
        var userId = User.GetUserIdOrThrow();

        var u = await _db.Users.FirstOrDefaultAsync(x => x.Id == userId, ct);
        if (u is null) return NotFound();

        _db.Users.Remove(u);
        await _db.SaveChangesAsync(ct);

        return NoContent();
    }
}
