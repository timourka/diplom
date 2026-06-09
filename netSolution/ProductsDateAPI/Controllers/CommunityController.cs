using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using PostgreSQLRepository;

namespace ProductsDateAPI.Controllers;

[ApiController]
[Route("api/community")]
public class CommunityController : ControllerBase
{
    private readonly AppDbContext _db;

    public CommunityController(AppDbContext db)
    {
        _db = db;
    }

    [HttpGet("top-contributors")]
    public async Task<ActionResult<List<TopContributorResponse>>> TopContributors(
        [FromQuery] int limit = 20,
        CancellationToken ct = default)
    {
        limit = Math.Clamp(limit, 1, 100);

        var rows = await (
                from report in _db.ErrorReports.AsNoTracking()
                join user in _db.Users.AsNoTracking() on report.UserId equals user.Id
                where report.Approved
                group report by new { user.Id, user.Email } into g
                select new
                {
                    UserId = g.Key.Id,
                    Login = g.Key.Email,
                    ApprovedReportsCount = g.Count()
                }
            )
            .OrderByDescending(x => x.ApprovedReportsCount)
            .ThenBy(x => x.Login)
            .Take(limit)
            .ToListAsync(ct);

        return Ok(rows
            .Select(x => new TopContributorResponse(
                x.UserId,
                x.Login,
                x.ApprovedReportsCount))
            .ToList());
    }
}

public record TopContributorResponse(
    int UserId,
    string Login,
    int ApprovedReportsCount);
