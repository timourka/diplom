using System.Net;
using System.Net.Http.Headers;
using System.Net.Http;
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

    public event Func<Task<string?>>? ReauthenticationRequested;

    public AdminApiClient(string baseUrl, string jwtToken)
    {
        _http = new HttpClient
        {
            BaseAddress = new Uri(baseUrl),
            Timeout = TimeSpan.FromMinutes(30)
        };

        SetAccessToken(jwtToken);
    }

    public void SetAccessToken(string jwtToken)
    {
        _http.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", jwtToken);
    }

    public void ClearAccessToken()
    {
        _http.DefaultRequestHeaders.Authorization = null;
    }

    public async Task<List<AdminErrorReportListItemDto>> GetErrorReportsAsync(CancellationToken ct = default)
    {
        var result = await GetFromJsonWithAuthRetryAsync<List<AdminErrorReportListItemDto>>(
            "api/admin/error-reports", ct);

        return result ?? new List<AdminErrorReportListItemDto>();
    }

    public async Task<AdminErrorReportDetailsDto?> GetErrorReportAsync(int reportId, CancellationToken ct = default)
    {
        return await GetFromJsonWithAuthRetryAsync<AdminErrorReportDetailsDto>(
            $"api/admin/error-reports/{reportId}", ct);
    }

    public async Task<Image?> GetFrameImageAsync(int reportId, int frameIndex, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Get, $"api/admin/error-reports/{reportId}/frames/{frameIndex}"),
            ct);

        if (!response.IsSuccessStatusCode)
            return null;

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var sourceImage = Image.FromStream(stream);
        return new Bitmap(sourceImage);
    }

    public async Task<List<YoloBboxDto>> GetFrameBboxesAsync(int reportId, int frameIndex, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Get, $"api/admin/error-reports/{reportId}/frames/{frameIndex}/bboxes"),
            ct);

        if (response.IsSuccessStatusCode)
        {
            await using var stream = await response.Content.ReadAsStreamAsync(ct);
            var bboxes = await JsonSerializer.DeserializeAsync<List<YoloBboxDto>>(stream, _jsonOptions, ct);
            return bboxes ?? new List<YoloBboxDto>();
        }

        // Fallback для старой версии API, где возвращалась только одна рамка.
        using var fallbackResponse = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Get, $"api/admin/error-reports/{reportId}/frames/{frameIndex}/bbox"),
            ct);

        if (!fallbackResponse.IsSuccessStatusCode)
            return new List<YoloBboxDto>();

        await using var fallbackStream = await fallbackResponse.Content.ReadAsStreamAsync(ct);
        var bbox = await JsonSerializer.DeserializeAsync<YoloBboxDto>(fallbackStream, _jsonOptions, ct);
        return bbox is null ? new List<YoloBboxDto>() : new List<YoloBboxDto> { bbox };
    }

    public async Task SetReportApprovedAsync(int reportId, bool approved, CancellationToken ct = default)
    {
        var body = JsonSerializer.Serialize(new ApproveErrorReportRequest(approved));
        using var response = await SendWithAuthRetryAsync(
            () => CreateJsonRequest(HttpMethod.Put, $"api/admin/error-reports/{reportId}/approve", body),
            ct);

        response.EnsureSuccessStatusCode();
    }

    public async Task DeleteReportAsync(int reportId, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Delete, $"api/admin/error-reports/{reportId}"),
            ct);

        response.EnsureSuccessStatusCode();
    }


    public async Task<List<AdminUserListItem>> GetUsersAsync(CancellationToken ct = default)
    {
        var result = await GetFromJsonWithAuthRetryAsync<List<AdminUserListItem>>("api/admin/users", ct);
        return result ?? new List<AdminUserListItem>();
    }

    public async Task<AdminUserDetailsDto?> GetUserAsync(int userId, CancellationToken ct = default)
        => await GetFromJsonWithAuthRetryAsync<AdminUserDetailsDto>($"api/admin/users/{userId}", ct);

    public async Task<AdminUserDetailsDto> UpdateUserAsync(int userId, AdminUserUpdateRequest request, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Put, $"api/admin/users/{userId}")
            {
                Content = JsonContent.Create(request)
            },
            ct);

        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync(ct);
            throw new HttpRequestException(string.IsNullOrWhiteSpace(error)
                ? $"Ошибка сохранения пользователя: {(int)response.StatusCode} {response.ReasonPhrase}"
                : error);
        }

        var result = await response.Content.ReadFromJsonAsync<AdminUserDetailsDto>(_jsonOptions, ct);
        return result ?? throw new Exception("Пустой ответ от сервера после сохранения пользователя.");
    }

    public async Task DeleteUserAsync(int userId, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Delete, $"api/admin/users/{userId}"),
            ct);

        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync(ct);
            throw new HttpRequestException(string.IsNullOrWhiteSpace(error)
                ? $"Ошибка удаления пользователя: {(int)response.StatusCode} {response.ReasonPhrase}"
                : error);
        }
    }

    public async Task BlockUserAsync(int userId, bool isBlocked, CancellationToken ct = default)
    {
        var body = JsonSerializer.Serialize(new SetBlockedRequest(isBlocked));
        using var response = await SendWithAuthRetryAsync(
            () => CreateJsonRequest(HttpMethod.Put, $"api/admin/users/{userId}/block", body),
            ct);

        response.EnsureSuccessStatusCode();
    }

    public async Task<TrainingJobStartResponse> StartTrainingAsync(StartTrainingRequest request, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Post, "api/admin/training/start")
            {
                Content = JsonContent.Create(request)
            },
            ct);

        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<TrainingJobStartResponse>(_jsonOptions, ct);
        return result ?? throw new Exception("Пустой ответ от сервера.");
    }

    public async Task<List<TrainingJobStatusResponse>> GetTrainingJobsAsync(CancellationToken ct = default)
    {
        var result = await GetFromJsonWithAuthRetryAsync<List<TrainingJobStatusResponse>>(
            "api/admin/training/jobs", ct);

        return result ?? new List<TrainingJobStatusResponse>();
    }

    public async Task<TrainingJobStatusResponse?> GetTrainingJobAsync(string jobId, CancellationToken ct = default)
        => await GetFromJsonWithAuthRetryAsync<TrainingJobStatusResponse>($"api/admin/training/jobs/{jobId}", ct);

    public async Task<TrainingJobStatusResponse> CancelTrainingJobAsync(string jobId, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Post, $"api/admin/training/jobs/{jobId}/cancel"),
            ct);

        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<TrainingJobStatusResponse>(_jsonOptions, ct);
        return result ?? throw new Exception("Пустой ответ от сервера при остановке задачи.");
    }


    public async Task<List<ModelVersionAdminResponse>> GetModelVersionsAsync(CancellationToken ct = default)
    {
        var result = await GetFromJsonWithAuthRetryAsync<List<ModelVersionAdminResponse>>(
            "api/admin/training/model-versions", ct);

        return result ?? new List<ModelVersionAdminResponse>();
    }

    public async Task<ModelVersionAdminResponse> PublishModelVersionAsync(int id, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Post, $"api/admin/training/model-versions/{id}/publish"),
            ct);

        response.EnsureSuccessStatusCode();
        return await ReadModelVersionResponseAsync(response, ct);
    }

    public async Task<ModelVersionAdminResponse> UnpublishModelVersionAsync(int id, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Post, $"api/admin/training/model-versions/{id}/unpublish"),
            ct);

        response.EnsureSuccessStatusCode();
        return await ReadModelVersionResponseAsync(response, ct);
    }

    public async Task<ModelVersionAdminResponse> PinModelVersionAsync(int id, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Post, $"api/admin/training/model-versions/{id}/pin"),
            ct);

        response.EnsureSuccessStatusCode();
        return await ReadModelVersionResponseAsync(response, ct);
    }

    public async Task<ModelVersionAdminResponse> UnpinModelVersionAsync(int id, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Post, $"api/admin/training/model-versions/{id}/unpin"),
            ct);

        response.EnsureSuccessStatusCode();
        return await ReadModelVersionResponseAsync(response, ct);
    }

    public async Task DeleteModelVersionAsync(int id, bool force = false, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Delete, $"api/admin/training/model-versions/{id}?force={force.ToString().ToLowerInvariant()}"),
            ct);

        response.EnsureSuccessStatusCode();
    }


    public async Task<string> ExportBackupAsync(string destinationPath, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Get, "api/admin/backup/export"),
            ct,
            HttpCompletionOption.ResponseHeadersRead);

        response.EnsureSuccessStatusCode();

        var fileName = response.Content.Headers.ContentDisposition?.FileNameStar
                       ?? response.Content.Headers.ContentDisposition?.FileName?.Trim('\"')
                       ?? Path.GetFileName(destinationPath);

        var directory = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        await using var source = await response.Content.ReadAsStreamAsync(ct);
        await using var destination = File.Create(destinationPath);
        await source.CopyToAsync(destination, ct);

        return fileName;
    }

    public async Task<BackupImportResultDto> ImportBackupAsync(string backupZipPath, bool replaceExisting, CancellationToken ct = default)
    {
        using var response = await SendWithAuthRetryAsync(
            () => CreateBackupImportRequest(backupZipPath, replaceExisting),
            ct);

        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync(ct);
            throw new HttpRequestException(string.IsNullOrWhiteSpace(error)
                ? $"Ошибка импорта backup: {(int)response.StatusCode} {response.ReasonPhrase}"
                : error);
        }

        var result = await response.Content.ReadFromJsonAsync<BackupImportResultDto>(_jsonOptions, ct);
        return result ?? throw new Exception("Пустой ответ от сервера после импорта backup.");
    }

    private async Task<T?> GetFromJsonWithAuthRetryAsync<T>(string requestUri, CancellationToken ct)
    {
        using var response = await SendWithAuthRetryAsync(
            () => new HttpRequestMessage(HttpMethod.Get, requestUri),
            ct);

        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        return await JsonSerializer.DeserializeAsync<T>(stream, _jsonOptions, ct);
    }

    private async Task<HttpResponseMessage> SendWithAuthRetryAsync(
        Func<HttpRequestMessage> requestFactory,
        CancellationToken ct,
        HttpCompletionOption completionOption = HttpCompletionOption.ResponseContentRead)
    {
        using var request = requestFactory();
        var response = await _http.SendAsync(request, completionOption, ct);

        if (!IsAuthorizationFailure(response.StatusCode))
            return response;

        response.Dispose();

        var refreshedToken = await RequestReauthenticationAsync();
        SetAccessToken(refreshedToken);

        using var retryRequest = requestFactory();
        response = await _http.SendAsync(retryRequest, completionOption, ct);

        if (IsAuthorizationFailure(response.StatusCode))
        {
            response.Dispose();
            throw new UnauthorizedAccessException("Вход выполнен, но сервер всё равно отказал в доступе. Проверьте, что используется административный профиль.");
        }

        return response;
    }

    private async Task<string> RequestReauthenticationAsync()
    {
        if (ReauthenticationRequested is null)
            throw new UnauthorizedAccessException("Сессия администратора истекла. Войдите повторно.");

        var token = await ReauthenticationRequested.Invoke();
        if (string.IsNullOrWhiteSpace(token))
            throw new UnauthorizedAccessException("Сессия администратора истекла, повторный вход не выполнен.");

        return token;
    }

    private static bool IsAuthorizationFailure(HttpStatusCode statusCode)
        => statusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden;

    private static HttpRequestMessage CreateJsonRequest(HttpMethod method, string requestUri, string body)
    {
        return new HttpRequestMessage(method, requestUri)
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json")
        };
    }

    private static HttpRequestMessage CreateBackupImportRequest(string backupZipPath, bool replaceExisting)
    {
        var fileStream = File.OpenRead(backupZipPath);
        var content = new MultipartFormDataContent();

        var fileContent = new StreamContent(fileStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/zip");

        content.Add(fileContent, "BackupZip", Path.GetFileName(backupZipPath));
        content.Add(new StringContent(replaceExisting.ToString().ToLowerInvariant()), "ReplaceExisting");

        return new HttpRequestMessage(HttpMethod.Post, "api/admin/backup/import")
        {
            Content = content
        };
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

        if (!auth.IsAdmin)
            throw new UnauthorizedAccessException("Этот профиль не является административным. Войдите под администратором.");

        return auth.AccessToken;
    }
}
