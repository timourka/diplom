import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

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
    try {
      final remoteInfo = await fetchLatestModelInfo();
      if (remoteInfo == null) {
        return ModelSyncResult(
          success: false,
          downloaded: false,
          message: 'На сервере пока нет опубликованной мобильной модели.',
          localInfo: await readLocalModelInfo(),
        );
      }

      final latestVersion = remoteInfo['modelVersionId'] as int?;
      if (latestVersion == null) {
        return ModelSyncResult(
          success: false,
          downloaded: false,
          message: 'Сервер вернул неполные данные о модели.',
          remoteInfo: remoteInfo,
          localInfo: await readLocalModelInfo(),
        );
      }

      final prefs = await SharedPreferences.getInstance();
      final currentVersion = prefs.getInt(_modelVersionKey);
      final currentPath = prefs.getString(_modelPathKey);
      final hasLocalFile = currentPath != null && await File(currentPath).exists();

      if (!force && currentVersion == latestVersion && hasLocalFile) {
        return ModelSyncResult(
          success: true,
          downloaded: false,
          message: 'Уже используется актуальная версия ИИ.',
          remoteInfo: remoteInfo,
          localInfo: await readLocalModelInfo(),
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

      final appDir = await getApplicationDocumentsDirectory();
      final modelsDir = Directory('${appDir.path}/models');
      if (!await modelsDir.exists()) {
        await modelsDir.create(recursive: true);
      }

      final format = (remoteInfo['mobileFormat']?.toString() ?? 'tflite').toLowerCase();
      final file = File('${modelsDir.path}/latest_model.$format');
      await file.writeAsBytes(modelResponse.bodyBytes, flush: true);

      await prefs.setInt(_modelVersionKey, latestVersion);
      await prefs.setString(_modelPathKey, file.path);
      await prefs.setString(_trainedAtKey, remoteInfo['trainedAt']?.toString() ?? '');
      await prefs.setString(_formatKey, format);
      await prefs.setString(_metricsJsonKey, remoteInfo['metricsJson']?.toString() ?? '');
      await prefs.setString(_sourceUrlKey, ApiClient.baseUrl);
      await prefs.setString(_syncedAtKey, DateTime.now().toIso8601String());

      return ModelSyncResult(
        success: true,
        downloaded: true,
        message: 'Модель ИИ обновлена до версии #$latestVersion.',
        remoteInfo: remoteInfo,
        localInfo: await readLocalModelInfo(),
      );
    } catch (e) {
      return ModelSyncResult(
        success: false,
        downloaded: false,
        message: 'Не удалось обновить ИИ: $e',
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

      return jsonDecode(metaResponse.body) as Map<String, dynamic>;
    } catch (_) {
      return null;
    }
  }

  Future<Map<String, dynamic>?> readLocalModelInfo() async {
    final prefs = await SharedPreferences.getInstance();
    final version = prefs.getInt(_modelVersionKey);
    final path = prefs.getString(_modelPathKey);

    if (version == null || path == null) {
      return null;
    }

    final file = File(path);
    final exists = await file.exists();

    return {
      'modelVersionId': version,
      'modelPath': path,
      'trainedAt': prefs.getString(_trainedAtKey),
      'mobileFormat': prefs.getString(_formatKey),
      'metricsJson': prefs.getString(_metricsJsonKey),
      'sourceUrl': prefs.getString(_sourceUrlKey),
      'syncedAt': prefs.getString(_syncedAtKey),
      'fileExists': exists,
      'fileSizeBytes': exists ? await file.length() : null,
    };
  }

  Future<String?> localModelPath() async {
    final prefs = await SharedPreferences.getInstance();
    final path = prefs.getString(_modelPathKey);
    if (path == null) {
      return null;
    }

    final file = File(path);
    return await file.exists() ? path : null;
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
