import 'dart:convert';
import 'dart:io';

import '../api/api_client.dart';
import '../storage/file_key_value_store.dart';

class PendingReportRepository {
  static const _pendingReportsKey = 'pending_error_reports_json';

  Future<List<Map<String, dynamic>>> readPendingReports() async {
    final raw = await FileKeyValueStore.getString(_pendingReportsKey);
    if (raw == null || raw.trim().isEmpty) return <Map<String, dynamic>>[];
    try {
      final decoded = jsonDecode(raw);
      if (decoded is! List) return <Map<String, dynamic>>[];
      return decoded
          .whereType<Map>()
          .map((x) => x.map((key, value) => MapEntry(key.toString(), value)))
          .toList(growable: true);
    } catch (_) {
      return <Map<String, dynamic>>[];
    }
  }

  Future<void> _writePendingReports(List<Map<String, dynamic>> items) async {
    await FileKeyValueStore.setString(_pendingReportsKey, jsonEncode(items));
  }

  Future<void> addPendingReport({
    required String zipPath,
    required String comment,
    String? validationToken,
    String? validationFrameName,
  }) async {
    final pending = await readPendingReports();
    pending.add({
      'zipPath': zipPath,
      'comment': comment,
      'validationToken': validationToken,
      'validationFrameName': validationFrameName,
      'createdAt': DateTime.now().toIso8601String(),
    });
    await _writePendingReports(pending);
  }

  Future<int> syncPendingReports(ApiClient api) async {
    final pending = await readPendingReports();
    if (pending.isEmpty) return 0;

    final remaining = <Map<String, dynamic>>[];
    var synced = 0;

    for (var i = 0; i < pending.length; i++) {
      final item = pending[i];
      final zipPath = item['zipPath']?.toString() ?? '';
      final comment = item['comment']?.toString() ?? '';
      final validationToken = item['validationToken']?.toString();
      final validationFrameName = item['validationFrameName']?.toString();
      if (zipPath.isEmpty || !await File(zipPath).exists()) {
        continue;
      }

      try {
        await api.uploadDatasetZip(
          zipPath,
          comment: comment,
          validationToken: validationToken,
          validationFrameName: validationFrameName,
        );
        synced++;
      } on NetworkApiException {
        remaining.add(item);
      } on AuthRequiredException {
        remaining.add(item);
        remaining.addAll(pending.skip(i + 1));
        await _writePendingReports(remaining);
        rethrow;
      } catch (_) {
        // Повреждённый архив или проваленная проверка не оставляем в вечной очереди.
      }
    }

    await _writePendingReports(remaining);
    return synced;
  }
}
