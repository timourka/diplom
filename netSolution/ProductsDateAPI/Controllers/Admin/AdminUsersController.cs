using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PostgreSQLRepository;
using ProductsDateAPI.Helpers;

namespace ProductsDateAPI.Controllers.Admin;

[ApiController]
[Route("api/admin/users")]
[Authorize(Policy = "AdminOnly")]
public class AdminUsersController : ControllerBase
{
    private readonly AppDbContext _db;

    public AdminUsersController(AppDbContext db) => _db = db;

    [HttpGet]
    public async Task<ActionResult<List<AdminUserListItem>>> GetAll(CancellationToken ct)
    {
        var items = await _db.Users
            .AsNoTracking()
            .OrderByDescending(x => x.Id)
            .Select(x => new AdminUserListItem(x.Id, x.Email, x.IsBlocked, x.IsAdmin, x.CreatedAt))
            .ToListAsync(ct);

        return Ok(items);
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<AdminUserListItem>> Get(int id, CancellationToken ct)
    {
        var u = await _db.Users.AsNoTracking().FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();
        return Ok(new AdminUserListItem(u.Id, u.Email, u.IsBlocked, u.IsAdmin, u.CreatedAt));
    }

    [HttpPut("{id:int}/block")]
    public async Task<ActionResult> SetBlocked(int id, SetBlockedRequest req, CancellationToken ct)
    {
        var u = await _db.Users.FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();

        if (u.IsAdmin && req.IsBlocked)
        {
            var hasAnotherActiveAdmin = await _db.Users
                .AnyAsync(x => x.Id != id && x.IsAdmin && !x.IsBlocked, ct);

            if (!hasAnotherActiveAdmin)
                return BadRequest("Нельзя заблокировать единственного активного администратора.");
        }

        u.IsBlocked = req.IsBlocked;
        await _db.SaveChangesAsync(ct);

        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<ActionResult> Delete(int id, CancellationToken ct)
    {
        var u = await _db.Users.FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();

        var currentUserId = User.GetUserIdOrThrow();
        if (u.Id == currentUserId)
            return BadRequest("Нельзя удалить собственный административный профиль.");

        if (u.IsAdmin)
        {
            var hasAnotherActiveAdmin = await _db.Users
                .AnyAsync(x => x.Id != id && x.IsAdmin && !x.IsBlocked, ct);

            if (!hasAnotherActiveAdmin)
                return BadRequest("Нельзя удалить единственного активного администратора.");
        }

        _db.Users.Remove(u);
        await _db.SaveChangesAsync(ct);

        return NoContent();
    }
}
