import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../auth/auth_flow.dart';
import '../auth/auth_state.dart';
import '../services/local_storage_repository.dart';

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
  final _localStorage = LocalStoredProductRepository();
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

  bool get _isExpired {
    final expiry = _expiry;
    if (expiry == null) return false;
    return _dateOnly(expiry.toLocal()).isBefore(_dateOnly(DateTime.now()));
  }

  Future<void> _pickDate() async {
    final today = _dateOnly(DateTime.now());
    final initial = _dateOnly(_expiry ?? today.add(const Duration(days: 7)));

    final firstDate = DateTime(2000, 1, 1);
    final defaultLastDate = today.add(const Duration(days: 3650));
    final lastDate = initial.isAfter(defaultLastDate) ? initial : defaultLastDate;

    DateTime selected = initial;

    final picked = await showModalBottomSheet<DateTime>(
      context: context,
      useRootNavigator: true,
      isScrollControlled: true,
      builder: (sheetContext) {
        return SafeArea(
          child: SizedBox(
            height: 390,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 10, 16, 6),
                  child: Row(
                    children: [
                      TextButton(
                        onPressed: () => Navigator.pop(sheetContext),
                        child: const Text(
                          'Отмена',
                          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
                        ),
                      ),
                      const Spacer(),
                      const Text(
                        'Срок годности',
                        style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
                      ),
                      const Spacer(),
                      TextButton(
                        onPressed: () => Navigator.pop(sheetContext, selected),
                        child: const Text(
                          'Готово',
                          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
                        ),
                      ),
                    ],
                  ),
                ),
                const Divider(height: 1),
                Expanded(
                  child: CupertinoTheme(
                    data: const CupertinoThemeData(
                      textTheme: CupertinoTextThemeData(
                        dateTimePickerTextStyle: TextStyle(
                          fontSize: 30,
                          fontWeight: FontWeight.w700,
                          color: Colors.black,
                        ),
                      ),
                    ),
                    child: CupertinoDatePicker(
                      mode: CupertinoDatePickerMode.date,
                      dateOrder: DatePickerDateOrder.dmy,
                      initialDateTime: initial,
                      minimumDate: firstDate,
                      maximumDate: lastDate,
                      minimumYear: firstDate.year,
                      maximumYear: lastDate.year,
                      itemExtent: 52,
                      onDateTimeChanged: (value) {
                        selected = _dateOnly(value);
                      },
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );

    if (picked != null) setState(() => _expiry = _dateOnly(picked));
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
      await AuthFlow.runWithReauth<void>(
        context: context,
        auth: widget.auth,
        after: 'manual',
        action: () async {
          final api = ApiClient(token: widget.auth.token);
          await api.addStoredProductByName(productName, _expiry!);
        },
      );
      if (!mounted) return;
      Navigator.pop(context, true);
    } on NetworkApiException {
      await _localStorage.addPendingProduct(productName, _expiry!);
      if (!mounted) return;
      Navigator.pop(context, true);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Интернета нет. Продукт сохранён локально и отправится позже.')),
      );
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.of(context).viewInsets.bottom;
    final dateText = _expiry == null
        ? 'Выбрать срок годности'
        : 'Срок годности: ${_expiry!.toLocal().toString().split(' ').first}';

    return Padding(
      padding: EdgeInsets.fromLTRB(16, 16, 16, 16 + pad),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Добавить продукт',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
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
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
                backgroundColor: _isExpired ? Colors.red.withOpacity(0.08) : null,
                foregroundColor: _isExpired ? Colors.red.shade900 : null,
                side: _isExpired ? BorderSide(color: Colors.red.shade300) : null,
              ),
              onPressed: _saving ? null : _pickDate,
              icon: const Icon(Icons.calendar_month, size: 28),
              label: Text(
                dateText,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
              ),
            ),
            if (_isExpired) ...[
              const SizedBox(height: 6),
              Text(
                'Эта дата уже прошла. Проверь срок годности перед сохранением.',
                style: TextStyle(color: Colors.red.shade800),
              ),
            ],
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
