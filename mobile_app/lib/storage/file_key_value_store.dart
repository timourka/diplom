import 'dart:convert';
import 'dart:io';

/// Локальное хранилище без platform plugins.
///
/// Важно: не использует shared_preferences/path_provider, потому что на части
/// Android-сборок они падали с Pigeon channel-error.
class FileKeyValueStore {
  static const String _androidPackageId = 'com.example.mobile_app';
  static Map<String, dynamic>? _cache;

  static Future<Directory> appDataDirectory() async {
    if (Platform.isAndroid) {
      final candidates = <String>[
        '/data/user/0/$_androidPackageId/files',
        '/data/data/$_androidPackageId/files',
      ];

      for (final path in candidates) {
        try {
          final dir = Directory(path);
          if (!await dir.exists()) {
            await dir.create(recursive: true);
          }
          if (await dir.exists()) {
            return dir;
          }
        } catch (_) {
          // Пробуем следующий вариант.
        }
      }
    }

    // Резерв для debug/desktop. Directory.systemTemp берётся из dart:io и не
    // требует platform channel.
    final fallback = Directory('${Directory.systemTemp.path}/mobile_app_state');
    if (!await fallback.exists()) {
      await fallback.create(recursive: true);
    }
    return fallback;
  }

  static Future<Directory> namedDirectory(String name) async {
    final safeName = name.replaceAll(RegExp(r'[^a-zA-Z0-9_\-]'), '_');
    final root = await appDataDirectory();
    final dir = Directory('${root.path}/$safeName');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  static Future<File> _stateFile() async {
    final dir = await appDataDirectory();
    return File('${dir.path}/app_state.json');
  }

  static Future<Map<String, dynamic>> _readAll() async {
    final cached = _cache;
    if (cached != null) {
      return Map<String, dynamic>.from(cached);
    }

    try {
      final file = await _stateFile();
      if (!await file.exists()) {
        _cache = <String, dynamic>{};
        return <String, dynamic>{};
      }

      final raw = await file.readAsString();
      if (raw.trim().isEmpty) {
        _cache = <String, dynamic>{};
        return <String, dynamic>{};
      }

      final decoded = jsonDecode(raw);
      if (decoded is Map<String, dynamic>) {
        _cache = decoded;
        return Map<String, dynamic>.from(decoded);
      }
    } catch (_) {
      // Если файл повреждён или временно недоступен, не ломаем приложение.
    }

    _cache = <String, dynamic>{};
    return <String, dynamic>{};
  }

  static Future<void> _writeAll(Map<String, dynamic> data) async {
    _cache = Map<String, dynamic>.from(data);

    try {
      final file = await _stateFile();
      await file.writeAsString(
        const JsonEncoder.withIndent('  ').convert(data),
        flush: true,
      );
    } catch (_) {
      // Запись не должна ронять приложение. В худшем случае значение будет
      // жить только в памяти до перезапуска.
    }
  }

  static Future<String?> getString(String key) async {
    final data = await _readAll();
    final value = data[key];
    return value == null ? null : value.toString();
  }

  static Future<int?> getInt(String key) async {
    final data = await _readAll();
    final value = data[key];
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '');
  }

  static Future<void> setString(String key, String value) async {
    final data = await _readAll();
    data[key] = value;
    await _writeAll(data);
  }

  static Future<void> setInt(String key, int value) async {
    final data = await _readAll();
    data[key] = value;
    await _writeAll(data);
  }

  static Future<void> remove(String key) async {
    final data = await _readAll();
    data.remove(key);
    await _writeAll(data);
  }
}
