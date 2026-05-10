import 'package:flutter/material.dart';

import '../api/api_config.dart';
import '../api/model_sync_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _modelSync = ModelSyncService();
  final _baseUrlController = TextEditingController();

  Map<String, dynamic>? _remoteInfo;
  Map<String, dynamic>? _localInfo;
  String? _status;
  bool _loading = true;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _baseUrlController.text = ApiConfig.baseUrl;
    _loadInfo();
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    super.dispose();
  }

  Future<void> _loadInfo() async {
    setState(() {
      _loading = true;
      _status = null;
    });

    final remote = await _modelSync.fetchLatestModelInfo();
    final local = await _modelSync.readLocalModelInfo();

    if (!mounted) return;
    setState(() {
      _remoteInfo = remote;
      _localInfo = local;
      _loading = false;
    });
  }

  Future<void> _saveBaseUrl() async {
    setState(() => _busy = true);

    try {
      final normalized = ApiConfig.normalizeForDisplay(_baseUrlController.text);
      await ApiConfig.setBaseUrl(normalized);
      _baseUrlController.text = ApiConfig.baseUrl;
      await _loadInfo();
      if (!mounted) return;
      setState(() => _status = 'Адрес сервера сохранён.');
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = 'Не удалось сохранить адрес: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _resetBaseUrl() async {
    setState(() => _busy = true);

    await ApiConfig.reset();
    _baseUrlController.text = ApiConfig.baseUrl;
    await _loadInfo();

    if (!mounted) return;
    setState(() {
      _status = 'Возвращён адрес по умолчанию.';
      _busy = false;
    });
  }

  Future<void> _updateAiVersion() async {
    setState(() {
      _busy = true;
      _status = 'Проверяю новую версию ИИ...';
    });

    final result = await _modelSync.trySyncLatestModel(force: true);
    final local = await _modelSync.readLocalModelInfo();

    if (!mounted) return;
    setState(() {
      _remoteInfo = result.remoteInfo ?? _remoteInfo;
      _localInfo = local;
      _status = result.message;
      _busy = false;
    });
  }

  Widget _infoTile(String title, String value) {
    return ListTile(
      dense: true,
      contentPadding: EdgeInsets.zero,
      title: Text(title),
      subtitle: Text(value.isEmpty ? '—' : value),
    );
  }

  Widget _buildAiInfoCard({
    required String title,
    required Map<String, dynamic>? info,
    required bool local,
  }) {
    if (info == null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(local
              ? 'Локальная модель ещё не загружена на устройство.'
              : 'На сервере пока нет опубликованной мобильной модели или сервер недоступен.'),
        ),
      );
    }

    final metrics = _modelSync.prettyMetrics(info['metricsJson']?.toString());

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            _infoTile('Версия', '#${info['modelVersionId']?.toString() ?? '—'}'),
            _infoTile('Формат', info['mobileFormat']?.toString() ?? ''),
            _infoTile('Обучена', info['trainedAt']?.toString() ?? ''),
            if (local) ...[
              _infoTile('Путь к файлу', info['modelPath']?.toString() ?? ''),
              _infoTile('Источник', info['sourceUrl']?.toString() ?? ''),
              _infoTile('Синхронизирована', info['syncedAt']?.toString() ?? ''),
              _infoTile('Файл существует', (info['fileExists'] ?? false) ? 'Да' : 'Нет'),
              _infoTile('Размер файла', '${info['fileSizeBytes']?.toString() ?? '—'} байт'),
            ],
            const SizedBox(height: 8),
            const Text('Метрики / справка', style: TextStyle(fontWeight: FontWeight.w600)),
            const SizedBox(height: 6),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(12),
              ),
              child: SelectableText(metrics),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Настройки'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _busy ? null : _loadInfo,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                TextField(
                  controller: _baseUrlController,
                  decoration: const InputDecoration(
                    labelText: 'BaseUrl сервера',
                    hintText: 'http://111.88.146.2:5099',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 12,
                  runSpacing: 12,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _busy ? null : _saveBaseUrl,
                      icon: const Icon(Icons.save),
                      label: const Text('Сохранить'),
                    ),
                    OutlinedButton.icon(
                      onPressed: _busy ? null : _resetBaseUrl,
                      icon: const Icon(Icons.restore),
                      label: const Text('По умолчанию'),
                    ),
                    FilledButton.icon(
                      onPressed: _busy ? null : _updateAiVersion,
                      icon: const Icon(Icons.system_update_alt),
                      label: const Text('Обновить версию ИИ'),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                if (_status != null)
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Text(_status!),
                    ),
                  ),
                const SizedBox(height: 8),
                _buildAiInfoCard(
                  title: 'Текущая версия ИИ на устройстве',
                  info: _localInfo,
                  local: true,
                ),
                const SizedBox(height: 12),
                _buildAiInfoCard(
                  title: 'Последняя версия ИИ на сервере',
                  info: _remoteInfo,
                  local: false,
                ),
              ],
            ),
    );
  }
}
