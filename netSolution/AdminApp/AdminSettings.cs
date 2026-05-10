using System.Text.Json;

namespace AdminApp;

public sealed class AdminSettings
{
    private const string SettingsFileName = "Settings.json";

    public string ApiBaseUrl { get; set; } = "http://localhost:5099/";

    public static AdminSettings Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, SettingsFileName);

        if (!File.Exists(path))
        {
            var defaultSettings = new AdminSettings();
            defaultSettings.Save(path);
            return defaultSettings;
        }

        var json = File.ReadAllText(path);
        var settings = JsonSerializer.Deserialize<AdminSettings>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        }) ?? new AdminSettings();

        settings.ApiBaseUrl = NormalizeBaseUrl(settings.ApiBaseUrl);
        return settings;
    }

    private void Save(string path)
    {
        var json = JsonSerializer.Serialize(this, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        File.WriteAllText(path, json);
    }

    private static string NormalizeBaseUrl(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new InvalidOperationException($"В {SettingsFileName} не указан ApiBaseUrl.");

        var trimmed = value.Trim();

        if (!Uri.TryCreate(trimmed, UriKind.Absolute, out var uri) ||
            (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
        {
            throw new InvalidOperationException(
                $"Некорректный ApiBaseUrl в {SettingsFileName}: '{value}'. Пример: http://111.88.146.2:5099/");
        }

        return trimmed.EndsWith('/') ? trimmed : trimmed + "/";
    }
}
