import 'dart:convert';

import 'package:flutter/services.dart';

import '../models/stored_product.dart';
import '../storage/file_key_value_store.dart';

class ExpiryNotificationService {
  static const _channel = MethodChannel('productsdate/local_notifications');
  static const _notifiedKey = 'expiry_notifications_shown_json';

  DateTime _dateOnly(DateTime value) => DateTime(value.year, value.month, value.day);

  Future<Map<String, String>> _readShown() async {
    final raw = await FileKeyValueStore.getString(_notifiedKey);
    if (raw == null || raw.trim().isEmpty) return <String, String>{};
    try {
      final decoded = jsonDecode(raw);
      if (decoded is Map) {
        return decoded.map((key, value) => MapEntry(key.toString(), value.toString()));
      }
    } catch (_) {}
    return <String, String>{};
  }

  Future<void> _writeShown(Map<String, String> shown) async {
    await FileKeyValueStore.setString(_notifiedKey, jsonEncode(shown));
  }

  Future<void> notifyDueToday(List<StoredProduct> items) async {
    final today = _dateOnly(DateTime.now());
    final todayKey = today.toIso8601String().split('T').first;
    final shown = await _readShown();
    var changed = false;

    for (final item in items) {
      final expiry = item.expiryAt;
      if (expiry == null) continue;
      if (_dateOnly(expiry.toLocal()) != today) continue;

      final key = item.id.toString();
      if (shown[key] == todayKey) continue;

      final name = item.product?.name.trim().isNotEmpty == true ? item.product!.name : 'Продукт';
      try {
        await _channel.invokeMethod('showExpiryNotification', {
          'id': item.id.abs(),
          'title': 'Сегодня истекает срок годности',
          'body': name,
        });
        shown[key] = todayKey;
        changed = true;
      } catch (_) {
        // Если уведомления недоступны, интерфейс не должен ломаться.
      }
    }

    if (changed) {
      await _writeShown(shown);
    }
  }
}
