import 'package:flutter/material.dart';
import '../auth/auth_state.dart';
import '../api/api_client.dart';

class ManualAddSheet extends StatefulWidget {
  final AuthState auth;
  const ManualAddSheet({super.key, required this.auth});

  @override
  State<ManualAddSheet> createState() => _ManualAddSheetState();
}

class _ManualAddSheetState extends State<ManualAddSheet> {
  List<dynamic> _products = [];
  int? _productId;
  DateTime? _expiry;

  String? _error;
  bool _loading = true;

  Future<void> _loadProducts() async {
    try {
      final api = ApiClient(token: widget.auth.token);
      _products = await api.products();
    } catch (e) {
      _error = e.toString();
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void initState() {
    super.initState();
    _loadProducts();
  }

  Future<void> _pickDate() async {
    final now = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      firstDate: now.subtract(const Duration(days: 1)),
      lastDate: now.add(const Duration(days: 3650)),
      initialDate: _expiry ?? now.add(const Duration(days: 7)),
    );
    if (picked != null) setState(() => _expiry = picked);
  }

  Future<void> _save() async {
    if (_productId == null) {
      setState(() => _error = 'Выбери продукт');
      return;
    }

    setState(() => _error = null);

    try {
      final api = ApiClient(token: widget.auth.token);
      await api.addStoredProduct(_productId!, _expiry);
      if (mounted) Navigator.pop(context);
    } catch (e) {
      setState(() => _error = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.of(context).viewInsets.bottom;

    return Padding(
      padding: EdgeInsets.fromLTRB(16, 16, 16, 16 + pad),
      child: _loading
          ? const SizedBox(height: 220, child: Center(child: CircularProgressIndicator()))
          : Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Ручное добавление',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),

                if (_error != null) ...[
                  Text(_error!, style: const TextStyle(color: Colors.red)),
                  const SizedBox(height: 8),
                ],

                DropdownButtonFormField<int>(
                  decoration: const InputDecoration(labelText: 'Продукт'),
                  items: _products.map((p) {
                    return DropdownMenuItem<int>(
                      value: p['id'] as int,
                      child: Text(p['name']?.toString() ?? 'Product'),
                    );
                  }).toList(),
                  onChanged: (v) => setState(() => _productId = v),
                ),

                const SizedBox(height: 12),

                Row(
                  children: [
                    Expanded(
                      child: Text(
                        _expiry == null
                            ? 'Срок годности: не выбран'
                            : 'Срок годности: ${_expiry!.toLocal().toString().split(" ").first}',
                      ),
                    ),
                    TextButton(
                      onPressed: _pickDate,
                      child: const Text('Выбрать дату'),
                    ),
                  ],
                ),

                const SizedBox(height: 12),

                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton(
                        onPressed: _save,
                        child: const Text('Сохранить'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
    );
  }
}
