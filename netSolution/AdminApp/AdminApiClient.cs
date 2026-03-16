using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using Contracts.Dtos;

namespace AdminApp;

public class AdminApiClient
{
    private readonly HttpClient _http;
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public AdminApiClient(string baseUrl, string jwtToken)
    {
        _http = new HttpClient
        {
            BaseAddress = new Uri(baseUrl)
        };

        _http.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", jwtToken);
    }

    public async Task<List<AdminErrorReportListItemDto>> GetErrorReportsAsync(CancellationToken ct = default)
    {
        var result = await _http.GetFromJsonAsync<List<AdminErrorReportListItemDto>>(
            "api/admin/error-reports", _jsonOptions, ct);

        return result ?? new List<AdminErrorReportListItemDto>();
    }

    public async Task<AdminErrorReportDetailsDto?> GetErrorReportAsync(int reportId, CancellationToken ct = default)
    {
        return await _http.GetFromJsonAsync<AdminErrorReportDetailsDto>(
            $"api/admin/error-reports/{reportId}", _jsonOptions, ct);
    }

    public async Task<Image?> GetFrameImageAsync(int reportId, int frameIndex, CancellationToken ct = default)
    {
        using var response = await _http.GetAsync(
            $"api/admin/error-reports/{reportId}/frames/{frameIndex}", ct);

        if (!response.IsSuccessStatusCode)
            return null;

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return Image.FromStream(stream);
    }

    public async Task<YoloBboxDto?> GetFrameBboxAsync(int reportId, int frameIndex, CancellationToken ct = default)
    {
        using var response = await _http.GetAsync(
            $"api/admin/error-reports/{reportId}/frames/{frameIndex}/bbox", ct);

        if (!response.IsSuccessStatusCode)
            return null;

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return await JsonSerializer.DeserializeAsync<YoloBboxDto>(stream, _jsonOptions, ct);
    }

    public async Task SetReportApprovedAsync(int reportId, bool approved, CancellationToken ct = default)
    {
        var body = JsonSerializer.Serialize(new ApproveErrorReportRequest(approved));
        using var content = new StringContent(body, Encoding.UTF8, "application/json");

        using var response = await _http.PutAsync(
            $"api/admin/error-reports/{reportId}/approve", content, ct);

        response.EnsureSuccessStatusCode();
    }

    public async Task DeleteReportAsync(int reportId, CancellationToken ct = default)
    {
        using var response = await _http.DeleteAsync(
            $"api/admin/error-reports/{reportId}", ct);

        response.EnsureSuccessStatusCode();
    }

    public async Task BlockUserAsync(int userId, bool isBlocked, CancellationToken ct = default)
    {
        var body = JsonSerializer.Serialize(new SetBlockedRequest(isBlocked));
        using var content = new StringContent(body, Encoding.UTF8, "application/json");

        using var response = await _http.PutAsync(
            $"api/admin/users/{userId}/block", content, ct);

        response.EnsureSuccessStatusCode();
    }

    public static async Task<string> LoginAsync(string baseUrl, string email, string password, CancellationToken ct = default)
    {
        using var http = new HttpClient { BaseAddress = new Uri(baseUrl) };

        var body = JsonSerializer.Serialize(new LoginRequest(email, password));
        using var content = new StringContent(body, Encoding.UTF8, "application/json");

        using var response = await http.PostAsync("api/Auth/login", content, ct);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        var auth = await JsonSerializer.DeserializeAsync<AuthResponse>(stream, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        }, ct);

        if (auth == null || string.IsNullOrWhiteSpace(auth.AccessToken))
            throw new Exception("Не удалось получить токен.");

        return auth.AccessToken;
    }
}