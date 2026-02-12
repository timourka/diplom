import 'package:flutter/material.dart';
import '../auth/auth_state.dart';
import '../api/api_client.dart';
import '../widgets/manual_add_sheet.dart';

class CabinetScreen extends StatefulWidget {
  final AuthState auth;
  const CabinetScreen({super.key, required this.auth});

  @override
  State<CabinetScreen> createState() => _CabinetScreenState();
}

class _CabinetScreenState extends State<CabinetScreen> {
  Map<String, dynamic>? _me;
  List<dynamic> _storage = [];
  String? _error;
  bool _loading = true;

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final api = ApiClient(token: widget.auth.token);
      _me = await api.me();
      _storage = await api.storage();
    } catch (e) {
      _error = e.toString();
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _openManualAdd() async {
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => ManualAddSheet(auth: widget.auth),
    );
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Личный кабинет'),
        actions: [
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
                    ExpansionTile(
                      title: const Text('Данные профиля'),
                      children: [
                        ListTile(
                          title: const Text('Email'),
                          subtitle: Text(_me?['email']?.toString() ?? ''),
                        ),
                        ListTile(
                          title: const Text('Id'),
                          subtitle: Text(_me?['id']?.toString() ?? ''),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'История сканирований / На хранении',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    if (_storage.isEmpty)
                      const Text('Пока пусто. Добавь продукт вручную (+).'),
                    ..._storage.map((x) {
                      final product = x['product'];
                      final name = product?['name']?.toString() ?? 'Product';
                      final expiry = x['expiryAt']?.toString() ?? '-';
                      return Card(
                        child: ListTile(
                          title: Text(name),
                          subtitle: Text('Expiry: $expiry'),
                        ),
                      );
                    }),
                  ],
                ),
    );
  }
}
