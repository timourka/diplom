import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../auth/auth_flow.dart';
import '../auth/auth_state.dart';
import '../models/stored_product.dart';
import '../services/expiry_notification_service.dart';
import '../services/local_storage_repository.dart';
import '../services/pending_report_repository.dart';
import '../widgets/manual_add_sheet.dart';
import 'settings_screen.dart';

class CabinetScreen extends StatefulWidget {
  final AuthState auth;
  const CabinetScreen({super.key, required this.auth});

  @override
  State<CabinetScreen> createState() => _CabinetScreenState();
}

class _CabinetScreenState extends State<CabinetScreen> {
  final _localStorage = LocalStoredProductRepository();
  final _pendingReports = PendingReportRepository();
  final _notifications = ExpiryNotificationService();

  Map<String, dynamic>? _me;
  List<StoredProduct> _storage = [];
  String? _error;
  String? _info;
  bool _loading = true;
  bool _offlineMode = false;

  DateTime _dateOnly(DateTime value) => DateTime(value.year, value.month, value.day);

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
      _info = null;
    });

    try {
      await AuthFlow.runWithReauth<void>(
        context: context,
        auth: widget.auth,
        after: 'profile',
        action: () async {
          final api = ApiClient(token: widget.auth.token);
          final syncedProducts = await _localStorage.syncPendingAdds(api);
          final syncedReports = await _pendingReports.syncPendingReports(api);
          _me = await api.me();
          final remoteStorage = await api.storage();
          await _localStorage.cacheStorage(remoteStorage);
          final merged = await _localStorage.mergeCachedWithPending(remoteStorage);
          _storage = _parseStorage(merged);
          _offlineMode = false;

          final parts = <String>[];
          if (syncedProducts > 0) parts.add('синхронизировано товаров: $syncedProducts');
          if (syncedReports > 0) parts.add('отправлено отчётов: $syncedReports');
          _info = parts.isEmpty ? null : parts.join(', ');
        },
      );
    } on NetworkApiException catch (e) {
      final cached = await _localStorage.readCachedStorage();
      final merged = await _localStorage.mergeCachedWithPending(cached);
      _storage = _parseStorage(merged);
      _offlineMode = true;
      _error = _storage.isEmpty ? e.toString() : null;
      _info = 'Сервер недоступен. Показаны локально сохранённые данные.';
    } catch (e) {
      _error = e.toString();
    } finally {
      if (mounted) {
        setState(() => _loading = false);
        await _notifications.notifyDueToday(_storage);
      }
    }
  }

  List<StoredProduct> _parseStorage(List<dynamic> raw) {
    return raw
        .whereType<Map>()
        .map((x) => StoredProduct.fromJson(x.map((key, value) => MapEntry(key.toString(), value))))
        .toList(growable: false);
  }

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _openManualAdd() async {
    final added = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      builder: (_) => ManualAddSheet(auth: widget.auth),
    );
    if (added == true) {
      await _load();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Продукт добавлен в личный кабинет')),
        );
      }
    }
  }

  Future<void> _deleteProduct(StoredProduct item) async {
    if (item.isPendingLocal) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Локально добавленный товар можно будет удалить после синхронизации.')),
      );
      return;
    }

    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Удалить продукт?'),
        content: Text('Удалить «${item.product?.name ?? 'продукт'}» из личного кабинета?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Отмена')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Удалить')),
        ],
      ),
    );
    if (ok != true) return;

    try {
      await AuthFlow.runWithReauth<void>(
        context: context,
        auth: widget.auth,
        after: 'profile',
        action: () => ApiClient(token: widget.auth.token).deleteStoredProduct(item.id),
      );
      await _load();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Продукт удалён')),
        );
      }
    } on NetworkApiException {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Удаление доступно только при подключении к серверу.')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Не удалось удалить: $e')));
    }
  }

  Future<void> _openSettings() async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const SettingsScreen()),
    );
    await _load();
  }

  Color? _cardColor(StoredProduct item) {
    final expiry = item.expiryAt;
    if (expiry == null) return null;

    final daysLeft = _dateOnly(expiry.toLocal()).difference(_dateOnly(DateTime.now())).inDays;
    if (daysLeft <= 7) return Colors.red.withOpacity(0.16);
    if (daysLeft <= 30) return Colors.orange.withOpacity(0.18);
    return null;
  }

  String _expiryText(StoredProduct item) {
    final expiry = item.expiryAt;
    if (expiry == null) return 'Срок годности: —';

    final date = expiry.toLocal().toString().split(' ').first;
    final daysLeft = _dateOnly(expiry.toLocal()).difference(_dateOnly(DateTime.now())).inDays;

    if (daysLeft < 0) return 'Срок годности: $date — истёк';
    if (daysLeft == 0) return 'Срок годности: $date — сегодня';
    return 'Срок годности: $date — осталось $daysLeft дн.';
  }

  Widget _buildProductTile(StoredProduct item) {
    final name = item.product?.name.trim().isNotEmpty == true ? item.product!.name : 'Без названия';

    return Card(
      color: _cardColor(item),
      child: ListTile(
        title: Text(
          name,
          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
        ),
        subtitle: Text(
          '${_expiryText(item)}${item.isPendingLocal ? '\nОжидает синхронизации' : ''}',
          style: const TextStyle(fontSize: 15),
        ),
        isThreeLine: item.isPendingLocal,
        trailing: IconButton(
          tooltip: 'Удалить',
          icon: const Icon(Icons.delete_outline),
          onPressed: _offlineMode || item.isPendingLocal ? null : () => _deleteProduct(item),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Личный кабинет'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _openSettings,
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _load,
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await widget.auth.logout();
              if (mounted) Navigator.pop(context);
            },
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _openManualAdd,
        child: const Icon(Icons.add),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : (_error != null)
              ? Center(child: Text(_error!, style: const TextStyle(color: Colors.red)))
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    if (_info != null) ...[
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(12),
                          child: Text(_info!),
                        ),
                      ),
                      const SizedBox(height: 8),
                    ],
                    ExpansionTile(
                      initiallyExpanded: false,
                      title: const Text('Данные профиля'),
                      children: [
                        ListTile(
                          title: const Text('Логин'),
                          subtitle: Text(_me?['email']?.toString() ?? widget.auth.login ?? ''),
                        ),
                        ListTile(
                          title: const Text('Настройки'),
                          subtitle: const Text('BaseUrl, версия ИИ и обновление модели'),
                          trailing: const Icon(Icons.chevron_right),
                          onTap: _openSettings,
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'На хранении',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    if (_storage.isEmpty)
                      const Text('Пока пусто. Добавь продукт вручную (+).'),
                    ..._storage.map(_buildProductTile),
                  ],
                ),
    );
  }
}
