import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import '../storage/file_key_value_store.dart';
import 'api_client.dart';

class ModelSyncResult {
  final bool success;
  final bool downloaded;
  final String message;
  final Map<String, dynamic>? remoteInfo;
  final Map<String, dynamic>? localInfo;

  const ModelSyncResult({
    required this.success,
    required this.downloaded,
    required this.message,
    this.remoteInfo,
    this.localInfo,
  });
}

class ModelSyncService {
  static const _modelVersionKey = 'mobile_model_version';
  static const _modelPathKey = 'mobile_model_path';
  static const _trainedAtKey = 'mobile_model_trained_at';
  static const _formatKey = 'mobile_model_format';
  static const _metricsJsonKey = 'mobile_model_metrics_json';
  static const _sourceUrlKey = 'mobile_model_source_url';
  static const _syncedAtKey = 'mobile_model_synced_at';

  Future<ModelSyncResult> trySyncLatestModel({bool force = false}) async {
    Map<String, dynamic>? remoteInfo;

    try {
      remoteInfo = await fetchLatestModelInfo();
      if (remoteInfo == null) {
        return ModelSyncResult(
          success: false,
          downloaded: false,
          message: 'На сервере пока нет опубликованной мобильной модели.',
          localInfo: await readLocalModelInfo(),
        );
      }

      final latestVersion = _asInt(remoteInfo['modelVersionId']);
      if (latestVersion == null) {
        return ModelSyncResult(
          success: false,
          downloaded: false,
          message: 'Сервер вернул неполные данные о модели.',
          remoteInfo: remoteInfo,
          localInfo: await readLocalModelInfo(),
        );
      }

      final localInfo = await readLocalModelInfo();
      final currentVersion = _asInt(localInfo?['modelVersionId']);
      final currentPath = localInfo?['modelPath']?.toString();
      final hasLocalFile = currentPath != null && await File(currentPath).exists();

      if (!force && currentVersion == latestVersion && hasLocalFile) {
        return ModelSyncResult(
          success: true,
          downloaded: false,
          message: 'Уже используется актуальная версия ИИ.',
          remoteInfo: remoteInfo,
          localInfo: localInfo,
        );
      }

      final modelResponse = await http.get(
        Uri.parse('${ApiClient.baseUrl}/api/mobile-models/latest/download'),
      );

      if (modelResponse.statusCode != 200) {
        return ModelSyncResult(
          success: false,
          downloaded: false,
          message: 'Не удалось скачать модель: HTTP ${modelResponse.statusCode}.',
          remoteInfo: remoteInfo,
          localInfo: await readLocalModelInfo(),
        );
      }

      final modelsDir = await _modelsDirectory();
      final format = (remoteInfo['mobileFormat']?.toString() ?? 'tflite').toLowerCase();
      final remoteFileName = remoteInfo['fileName']?.toString();
      final safeFileName = _safeModelFileName(remoteFileName, format);
      final file = File('${modelsDir.path}/$safeFileName');
      await file.writeAsBytes(modelResponse.bodyBytes, flush: true);

      await FileKeyValueStore.setInt(_modelVersionKey, latestVersion);
      await FileKeyValueStore.setString(_modelPathKey, file.path);
      await FileKeyValueStore.setString(_trainedAtKey, remoteInfo['trainedAt']?.toString() ?? '');
      await FileKeyValueStore.setString(_formatKey, format);
      await FileKeyValueStore.setString(_metricsJsonKey, remoteInfo['metricsJson']?.toString() ?? '');
      await FileKeyValueStore.setString(_sourceUrlKey, ApiClient.baseUrl);
      await FileKeyValueStore.setString(_syncedAtKey, DateTime.now().toIso8601String());

      final updatedLocalInfo = await readLocalModelInfo();

      return ModelSyncResult(
        success: true,
        downloaded: true,
        message: 'Модель ИИ обновлена до версии #$latestVersion.',
        remoteInfo: remoteInfo,
        localInfo: updatedLocalInfo,
      );
    } catch (e) {
      return ModelSyncResult(
        success: false,
        downloaded: false,
        message: 'Не удалось обновить ИИ: $e',
        remoteInfo: remoteInfo,
        localInfo: await readLocalModelInfo(),
      );
    }
  }

  Future<Map<String, dynamic>?> fetchLatestModelInfo() async {
    try {
      final metaResponse = await http.get(
        Uri.parse('${ApiClient.baseUrl}/api/mobile-models/latest'),
      );

      if (metaResponse.statusCode != 200) {
        return null;
      }

      final decoded = jsonDecode(metaResponse.body);
      return decoded is Map<String, dynamic> ? decoded : null;
    } catch (_) {
      return null;
    }
  }

  Future<Map<String, dynamic>?> readLocalModelInfo() async {
    try {
      final version = await FileKeyValueStore.getInt(_modelVersionKey);
      final path = await FileKeyValueStore.getString(_modelPathKey);

      if (version == null || path == null || path.isEmpty) {
        return null;
      }

      final file = File(path);
      final exists = await file.exists();

      return {
        'modelVersionId': version,
        'modelPath': path,
        'trainedAt': await FileKeyValueStore.getString(_trainedAtKey),
        'mobileFormat': await FileKeyValueStore.getString(_formatKey),
        'metricsJson': await FileKeyValueStore.getString(_metricsJsonKey),
        'sourceUrl': await FileKeyValueStore.getString(_sourceUrlKey),
        'syncedAt': await FileKeyValueStore.getString(_syncedAtKey),
        'fileExists': exists,
        'fileSizeBytes': exists ? await file.length() : null,
      };
    } catch (_) {
      return null;
    }
  }

  Future<String?> localModelPath() async {
    try {
      final path = await FileKeyValueStore.getString(_modelPathKey);
      if (path != null && path.isNotEmpty) {
        final file = File(path);
        if (await file.exists()) {
          return path;
        }
      }

      // Резервный вариант: если метаданные не сохранились, но файл модели есть,
      // берём самый свежий .tflite/.lite из папки models.
      final modelsDir = await _modelsDirectory();
      final files = await modelsDir
          .list()
          .where((entity) => entity is File)
          .cast<File>()
          .where((file) {
            final name = file.path.toLowerCase();
            return name.endsWith('.tflite') || name.endsWith('.lite');
          })
          .toList();

      if (files.isEmpty) {
        return null;
      }

      files.sort((a, b) => b.lastModifiedSync().compareTo(a.lastModifiedSync()));
      return files.first.path;
    } catch (_) {
      return null;
    }
  }

  Future<Directory> _modelsDirectory() {
    return FileKeyValueStore.namedDirectory('models');
  }

  int? _asInt(Object? value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '');
  }

  String _safeModelFileName(String? remoteFileName, String format) {
    final fallback = 'latest_model.$format';
    if (remoteFileName == null || remoteFileName.trim().isEmpty) {
      return fallback;
    }

    final cleaned = remoteFileName.split('/').last.split('\\').last.trim();
    if (cleaned.isEmpty || cleaned.contains('..')) {
      return fallback;
    }

    return 'published_$cleaned';
  }

  String prettyMetrics(String? raw) {
    if (raw == null || raw.trim().isEmpty) {
      return 'Метрики пока не сохранены.';
    }

    try {
      final decoded = jsonDecode(raw);
      final encoder = const JsonEncoder.withIndent('  ');
      return encoder.convert(decoded);
    } catch (_) {
      return raw;
    }
  }
}
