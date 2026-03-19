import 'package:shared_preferences/shared_preferences.dart';

class ApiConfig {
  static const defaultBaseUrl = 'http://10.0.2.2:5099';
  static const _baseUrlKey = 'api_base_url';

  static String _baseUrl = defaultBaseUrl;

  static String get baseUrl => _baseUrl;

  static Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl = _normalize(prefs.getString(_baseUrlKey) ?? defaultBaseUrl);
  }

  static Future<void> setBaseUrl(String value) async {
    _baseUrl = _normalize(value);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_baseUrlKey, _baseUrl);
  }

  static Future<void> reset() => setBaseUrl(defaultBaseUrl);

  static String normalizeForDisplay(String value) => _normalize(value);

  static String _normalize(String value) {
    var normalized = value.trim();
    if (normalized.isEmpty) {
      normalized = defaultBaseUrl;
    }

    if (!normalized.startsWith('http://') &&
        !normalized.startsWith('https://')) {
      normalized = 'http://$normalized';
    }

    while (normalized.endsWith('/')) {
      normalized = normalized.substring(0, normalized.length - 1);
    }

    return normalized;
  }
}
