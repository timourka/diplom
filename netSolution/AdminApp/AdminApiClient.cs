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

    public async Task<TrainingJobStartResponse> StartTrainingAsync(StartTrainingRequest request, CancellationToken ct = default)
    {
        using var response = await _http.PostAsJsonAsync("api/admin/training/start", request, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<TrainingJobStartResponse>(_jsonOptions, ct);
        return result ?? throw new Exception("Пустой ответ от сервера.");
    }

    public async Task<List<TrainingJobStatusResponse>> GetTrainingJobsAsync(CancellationToken ct = default)
    {
        var result = await _http.GetFromJsonAsync<List<TrainingJobStatusResponse>>(
            "api/admin/training/jobs", _jsonOptions, ct);

        return result ?? new List<TrainingJobStatusResponse>();
    }

    public async Task<TrainingJobStatusResponse?> GetTrainingJobAsync(string jobId, CancellationToken ct = default)
        => await _http.GetFromJsonAsync<TrainingJobStatusResponse>($"api/admin/training/jobs/{jobId}", _jsonOptions, ct);

    public async Task<TrainingJobStatusResponse> CancelTrainingJobAsync(string jobId, CancellationToken ct = default)
    {
        using var response = await _http.PostAsync($"api/admin/training/jobs/{jobId}/cancel", content: null, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<TrainingJobStatusResponse>(_jsonOptions, ct);
        return result ?? throw new Exception("Пустой ответ от сервера при остановке задачи.");
    }


    public async Task<List<ModelVersionAdminResponse>> GetModelVersionsAsync(CancellationToken ct = default)
    {
        var result = await _http.GetFromJsonAsync<List<ModelVersionAdminResponse>>(
            "api/admin/training/model-versions", _jsonOptions, ct);

        return result ?? new List<ModelVersionAdminResponse>();
    }

    public async Task<ModelVersionAdminResponse> PublishModelVersionAsync(int id, CancellationToken ct = default)
    {
        using var response = await _http.PostAsync($"api/admin/training/model-versions/{id}/publish", content: null, ct);
        response.EnsureSuccessStatusCode();
        return await ReadModelVersionResponseAsync(response, ct);
    }

    public async Task<ModelVersionAdminResponse> UnpublishModelVersionAsync(int id, CancellationToken ct = default)
    {
        using var response = await _http.PostAsync($"api/admin/training/model-versions/{id}/unpublish", content: null, ct);
        response.EnsureSuccessStatusCode();
        return await ReadModelVersionResponseAsync(response, ct);
    }

    public async Task<ModelVersionAdminResponse> PinModelVersionAsync(int id, CancellationToken ct = default)
    {
        using var response = await _http.PostAsync($"api/admin/training/model-versions/{id}/pin", content: null, ct);
        response.EnsureSuccessStatusCode();
        return await ReadModelVersionResponseAsync(response, ct);
    }

    public async Task<ModelVersionAdminResponse> UnpinModelVersionAsync(int id, CancellationToken ct = default)
    {
        using var response = await _http.PostAsync($"api/admin/training/model-versions/{id}/unpin", content: null, ct);
        response.EnsureSuccessStatusCode();
        return await ReadModelVersionResponseAsync(response, ct);
    }

    public async Task DeleteModelVersionAsync(int id, bool force = false, CancellationToken ct = default)
    {
        using var response = await _http.DeleteAsync($"api/admin/training/model-versions/{id}?force={force.ToString().ToLowerInvariant()}", ct);
        response.EnsureSuccessStatusCode();
    }

    private async Task<ModelVersionAdminResponse> ReadModelVersionResponseAsync(HttpResponseMessage response, CancellationToken ct)
    {
        var result = await response.Content.ReadFromJsonAsync<ModelVersionAdminResponse>(_jsonOptions, ct);
        return result ?? throw new Exception("Пустой ответ от сервера по версии модели.");
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
