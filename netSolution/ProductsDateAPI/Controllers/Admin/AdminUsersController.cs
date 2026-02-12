using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PostgreSQLRepository;

namespace ProductsDateAPI.Controllers.Admin;

[ApiController]
[Route("api/admin/users")]
// пока без ролей, но закрываем авторизацией
[Authorize]
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
            .Select(x => new AdminUserListItem(x.Id, x.Email, x.IsBlocked, x.CreatedAt))
            .ToListAsync(ct);

        return Ok(items);
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<AdminUserListItem>> Get(int id, CancellationToken ct)
    {
        var u = await _db.Users.AsNoTracking().FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();
        return Ok(new AdminUserListItem(u.Id, u.Email, u.IsBlocked, u.CreatedAt));
    }

    [HttpPut("{id:int}/block")]
    public async Task<ActionResult> SetBlocked(int id, SetBlockedRequest req, CancellationToken ct)
    {
        var u = await _db.Users.FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();

        u.IsBlocked = req.IsBlocked;
        await _db.SaveChangesAsync(ct);

        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<ActionResult> Delete(int id, CancellationToken ct)
    {
        var u = await _db.Users.FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();

        _db.Users.Remove(u);
        await _db.SaveChangesAsync(ct);

        return NoContent();
    }
}
