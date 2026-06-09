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
    public async Task<ActionResult<AdminUserDetailsDto>> Get(int id, CancellationToken ct)
    {
        var u = await _db.Users.AsNoTracking().FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();

        var storedProductsCount = await _db.StoredProducts.CountAsync(x => x.UserId == id, ct);
        var errorReportsCount = await _db.ErrorReports.CountAsync(x => x.UserId == id, ct);
        var approvedReportsCount = await _db.ErrorReports.CountAsync(x => x.UserId == id && x.Approved, ct);

        return Ok(new AdminUserDetailsDto(
            u.Id,
            u.Email,
            u.IsBlocked,
            u.IsAdmin,
            u.CreatedAt,
            u.SettingsJson,
            storedProductsCount,
            errorReportsCount,
            approvedReportsCount));
    }

    [HttpPut("{id:int}")]
    public async Task<ActionResult<AdminUserDetailsDto>> Update(int id, AdminUserUpdateRequest req, CancellationToken ct)
    {
        var u = await _db.Users.FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();

        var currentUserId = User.GetUserIdOrThrow();
        var login = (req.Login ?? string.Empty).Trim();

        if (string.IsNullOrWhiteSpace(login))
            return BadRequest("Логин не может быть пустым.");

        var loginExists = await _db.Users.AnyAsync(x => x.Id != id && x.Email == login, ct);
        if (loginExists)
            return Conflict("Пользователь с таким логином уже существует.");

        if (u.Id == currentUserId && req.IsBlocked)
            return BadRequest("Нельзя заблокировать собственный административный профиль.");

        if (u.Id == currentUserId && !req.IsAdmin)
            return BadRequest("Нельзя снять административные права с собственного профиля.");

        if (u.IsAdmin && (!req.IsAdmin || req.IsBlocked))
        {
            var hasAnotherActiveAdmin = await _db.Users
                .AnyAsync(x => x.Id != id && x.IsAdmin && !x.IsBlocked, ct);

            if (!hasAnotherActiveAdmin)
                return BadRequest("Нельзя оставить систему без активного администратора.");
        }

        u.Email = login;
        u.IsBlocked = req.IsBlocked;
        u.IsAdmin = req.IsAdmin;
        u.SettingsJson = string.IsNullOrWhiteSpace(req.SettingsJson) ? null : req.SettingsJson;

        await _db.SaveChangesAsync(ct);

        return await Get(id, ct);
    }

    [HttpPut("{id:int}/block")]
    public async Task<ActionResult> SetBlocked(int id, SetBlockedRequest req, CancellationToken ct)
    {
        var u = await _db.Users.FirstOrDefaultAsync(x => x.Id == id, ct);
        if (u is null) return NotFound();

        var currentUserId = User.GetUserIdOrThrow();
        if (u.Id == currentUserId && req.IsBlocked)
            return BadRequest("Нельзя заблокировать собственный административный профиль.");

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
