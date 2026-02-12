namespace ProductsDateAPI.Security;

public sealed class JwtOptions
{
    public string Issuer { get; set; } = "ProductsDateAPI";
    public string Audience { get; set; } = "ProductsDateAPI";
    public string Secret { get; set; } = "CHANGE_ME";
    public int AccessTokenMinutes { get; set; } = 60;
}
