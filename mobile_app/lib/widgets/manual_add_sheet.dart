import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../auth/auth_state.dart';

class ManualAddSheet extends StatefulWidget {
  final AuthState auth;
  final DateTime? initialExpiry;

  const ManualAddSheet({
    super.key,
    required this.auth,
    this.initialExpiry,
  });

  @override
  State<ManualAddSheet> createState() => _ManualAddSheetState();
}

class _ManualAddSheetState extends State<ManualAddSheet> {
  final _nameController = TextEditingController();
  DateTime? _expiry;

  String? _error;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _expiry = widget.initialExpiry;
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  DateTime _dateOnly(DateTime value) => DateTime(value.year, value.month, value.day);

  Future<void> _pickDate() async {
    final today = _dateOnly(DateTime.now());
    final initial = _dateOnly(_expiry ?? today.add(const Duration(days: 7)));

    // showDatePicker падает, если initialDate раньше firstDate.
    // OCR может распознать уже прошедшую дату, поэтому календарь должен уметь
    // открываться и для таких значений, чтобы пользователь мог исправить дату.
    final firstDate = DateTime(2000, 1, 1);
    final defaultLastDate = today.add(const Duration(days: 3650));
    final lastDate = initial.isAfter(defaultLastDate) ? initial : defaultLastDate;

    final picked = await showDatePicker(
      context: context,
      firstDate: firstDate,
      lastDate: lastDate,
      initialDate: initial,
    );
    if (picked != null) setState(() => _expiry = picked);
  }

  Future<void> _save() async {
    final productName = _nameController.text.trim();

    if (productName.isEmpty) {
      setState(() => _error = 'Введи название продукта');
      return;
    }

    if (_expiry == null) {
      setState(() => _error = 'Выбери срок годности');
      return;
    }

    setState(() {
      _error = null;
      _saving = true;
    });

    try {
      final api = ApiClient(token: widget.auth.token);
      await api.addStoredProductByName(productName, _expiry!);
      if (!mounted) return;
      Navigator.pop(context, true);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.of(context).viewInsets.bottom;

    return Padding(
      padding: EdgeInsets.fromLTRB(16, 16, 16, 16 + pad),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Добавить продукт',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            if (_error != null) ...[
              Text(_error!, style: const TextStyle(color: Colors.red)),
              const SizedBox(height: 8),
            ],
            TextField(
              controller: _nameController,
              textInputAction: TextInputAction.done,
              decoration: const InputDecoration(
                labelText: 'Название продукта',
                hintText: 'Например: молоко, йогурт, сыр',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: _saving ? null : _pickDate,
              icon: const Icon(Icons.calendar_month),
              label: Text(
                _expiry == null
                    ? 'Выбрать срок годности'
                    : 'Срок годности: ${_expiry!.toLocal().toString().split(' ').first}',
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: _saving ? null : _save,
              icon: _saving
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.save),
              label: Text(_saving ? 'Сохраняем...' : 'Добавить в личный кабинет'),
            ),
          ],
        ),
      ),
    );
  }
}
