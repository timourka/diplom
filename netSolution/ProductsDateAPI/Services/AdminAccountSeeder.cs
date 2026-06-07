using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Models.Entities;
using PostgreSQLRepository;

namespace ProductsDateAPI.Services;

public sealed class AdminAccountSeeder
{
    private readonly AppDbContext _db;
    private readonly AdminSeedOptions _options;
    private readonly ILogger<AdminAccountSeeder> _logger;

    public AdminAccountSeeder(
        AppDbContext db,
        IOptions<AdminSeedOptions> options,
        ILogger<AdminAccountSeeder> logger)
    {
        _db = db;
        _options = options.Value;
        _logger = logger;
    }

    public async Task SeedAsync(CancellationToken ct = default)
    {
        if (!_options.Enabled)
            return;

        var email = _options.Email?.Trim().ToLowerInvariant();
        var password = _options.Password;

        if (string.IsNullOrWhiteSpace(email) || string.IsNullOrWhiteSpace(password))
        {
            _logger.LogWarning(
                "Admin seed is enabled, but AdminSeed:Email or AdminSeed:Password is empty. Admin account was not created.");
            return;
        }

        var user = await _db.Users.FirstOrDefaultAsync(x => x.Email == email, ct);
        if (user is not null)
        {
            if (!user.IsAdmin)
            {
                user.IsAdmin = true;
                user.IsBlocked = false;
                await _db.SaveChangesAsync(ct);
                _logger.LogInformation("Existing user {Email} was promoted to admin.", email);
            }

            return;
        }

        var admin = new User
        {
            Email = email,
            PasswordHash = BCrypt.Net.BCrypt.HashPassword(password),
            IsBlocked = false,
            IsAdmin = true,
            CreatedAt = DateTime.UtcNow
        };

        _db.Users.Add(admin);
        await _db.SaveChangesAsync(ct);
        _logger.LogInformation("Admin account {Email} was created.", email);
    }
}
