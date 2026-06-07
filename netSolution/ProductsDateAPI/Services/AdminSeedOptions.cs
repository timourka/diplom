namespace ProductsDateAPI.Services;

public sealed class AdminSeedOptions
{
    public bool Enabled { get; set; } = true;
    public string? Email { get; set; }
    public string? Password { get; set; }
}
