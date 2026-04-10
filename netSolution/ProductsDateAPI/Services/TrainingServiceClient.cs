using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using Contracts.Dtos;
using Microsoft.Extensions.Options;
using ProductsDateAPI.Models;

namespace ProductsDateAPI.Services;

public class TrainingServiceClient
{
    private readonly HttpClient _http;
    private readonly TrainingServiceOptions _options;
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public TrainingServiceClient(HttpClient http, IOptions<TrainingServiceOptions> options)
    {
        _http = http;
        _options = options.Value;

        if (_http.BaseAddress is null)
            _http.BaseAddress = new Uri(_options.BaseUrl);

        if (!string.IsNullOrWhiteSpace(_options.ApiKey))
        {
            _http.DefaultRequestHeaders.Remove("X-Api-Key");
            _http.DefaultRequestHeaders.Add("X-Api-Key", _options.ApiKey);
        }
    }

    public async Task<TrainingJobStartResponse> StartTrainingAsync(
        string datasetZipPath,
        StartTrainingRequest? request,
        CancellationToken ct = default)
    {
        using var form = new MultipartFormDataContent();
        await using var fileStream = File.OpenRead(datasetZipPath);
        using var fileContent = new StreamContent(fileStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/zip");
        form.Add(fileContent, "datasetZip", Path.GetFileName(datasetZipPath));

        AddField(form, "baseModel", request?.BaseModel ?? "yolov8n.pt");
        AddField(form, "epochs", (request?.Epochs ?? 50).ToString());
        AddField(form, "imgsz", (request?.ImgSize ?? 640).ToString());
        AddField(form, "batch", (request?.Batch ?? 16).ToString());
        AddField(form, "device", string.IsNullOrWhiteSpace(request?.Device) ? "auto" : request!.Device!);
        AddField(form, "exportInt8", ((request?.ExportInt8) ?? true).ToString().ToLowerInvariant());
        AddField(form, "exportNms", ((request?.ExportNms) ?? true).ToString().ToLowerInvariant());
        AddField(form, "mobileFormat", request?.MobileFormat ?? "tflite");
        AddField(form, "quantizationFraction", ((request?.QuantizationFraction) ?? 0.3).ToString(System.Globalization.CultureInfo.InvariantCulture));

        using var response = await _http.PostAsync("jobs/train", form, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<TrainingJobStartResponse>(_jsonOptions, ct);
        return result ?? throw new InvalidOperationException("Training service returned empty response.");
    }

    public async Task<List<TrainingJobStatusResponse>> GetJobsAsync(CancellationToken ct = default)
    {
        var result = await _http.GetFromJsonAsync<List<TrainingJobStatusResponse>>("jobs", _jsonOptions, ct);
        return result ?? new List<TrainingJobStatusResponse>();
    }

    public async Task<TrainingJobStatusResponse?> GetJobAsync(string jobId, CancellationToken ct = default)
        => await _http.GetFromJsonAsync<TrainingJobStatusResponse>($"jobs/{jobId}", _jsonOptions, ct);

    public async Task<TrainingJobStatusResponse> CancelJobAsync(string jobId, CancellationToken ct = default)
    {
        using var response = await _http.PostAsync($"jobs/{jobId}/cancel", content: null, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<TrainingJobStatusResponse>(_jsonOptions, ct);
        return result ?? throw new InvalidOperationException("Training service returned empty response for cancel.");
    }

    public async Task<(byte[] Bytes, string? ContentType, string? FileName)> DownloadArtifactAsync(
        string jobId,
        string artifact,
        CancellationToken ct = default)
    {
        using var response = await _http.GetAsync($"jobs/{jobId}/artifacts/{artifact}", ct);
        response.EnsureSuccessStatusCode();

        var bytes = await response.Content.ReadAsByteArrayAsync(ct);
        var contentType = response.Content.Headers.ContentType?.MediaType;
        var fileName = response.Content.Headers.ContentDisposition?.FileNameStar
            ?? response.Content.Headers.ContentDisposition?.FileName;

        return (bytes, contentType, fileName?.Trim('"'));
    }

    private static void AddField(MultipartFormDataContent form, string key, string value)
        => form.Add(new StringContent(value), key);
}
