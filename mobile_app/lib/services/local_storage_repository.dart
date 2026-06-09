import 'dart:convert';

import '../api/api_client.dart';
import '../storage/file_key_value_store.dart';

class LocalStoredProductRepository {
  static const _cacheKey = 'storage_cache_json';
  static const _pendingAddsKey = 'storage_pending_adds_json';

  Future<List<dynamic>> readCachedStorage() async {
    final raw = await FileKeyValueStore.getString(_cacheKey);
    if (raw == null || raw.trim().isEmpty) return <dynamic>[];
    try {
      final decoded = jsonDecode(raw);
      return decoded is List ? decoded : <dynamic>[];
    } catch (_) {
      return <dynamic>[];
    }
  }

  Future<void> cacheStorage(List<dynamic> items) async {
    await FileKeyValueStore.setString(_cacheKey, jsonEncode(items));
  }

  Future<List<Map<String, dynamic>>> readPendingAdds() async {
    final raw = await FileKeyValueStore.getString(_pendingAddsKey);
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

  Future<void> _writePendingAdds(List<Map<String, dynamic>> items) async {
    await FileKeyValueStore.setString(_pendingAddsKey, jsonEncode(items));
  }

  Future<void> addPendingProduct(String productName, DateTime expiryAt) async {
    final pending = await readPendingAdds();
    final localId = -DateTime.now().millisecondsSinceEpoch;
    pending.add({
      'localId': localId,
      'productName': productName,
      'expiryAt': expiryAt.toIso8601String(),
      'createdAt': DateTime.now().toIso8601String(),
    });
    await _writePendingAdds(pending);
  }

  Future<int> syncPendingAdds(ApiClient api) async {
    final pending = await readPendingAdds();
    if (pending.isEmpty) return 0;

    final remaining = <Map<String, dynamic>>[];
    var synced = 0;

    for (final item in pending) {
      final name = item['productName']?.toString().trim() ?? '';
      final rawExpiry = item['expiryAt']?.toString();
      final expiry = rawExpiry == null ? null : DateTime.tryParse(rawExpiry);

      if (name.isEmpty || expiry == null) {
        continue;
      }

      try {
        await api.addStoredProductByName(name, expiry);
        synced++;
      } on NetworkApiException {
        remaining.add(item);
      } on AuthRequiredException {
        remaining.add(item);
        await _writePendingAdds(remaining..addAll(pending.skip(pending.indexOf(item) + 1)));
        rethrow;
      } catch (_) {
        // Некорректную локальную запись не гоняем бесконечно.
      }
    }

    await _writePendingAdds(remaining);
    return synced;
  }

  Future<List<dynamic>> mergeCachedWithPending(List<dynamic> cached) async {
    final pending = await readPendingAdds();
    if (pending.isEmpty) return cached;

    final result = List<dynamic>.from(cached);
    for (final p in pending) {
      result.insert(0, {
        'id': p['localId'],
        'isPendingLocal': true,
        'expiryAt': p['expiryAt'],
        'createdAt': p['createdAt'],
        'product': {'name': p['productName']},
      });
    }
    return result;
  }
}
