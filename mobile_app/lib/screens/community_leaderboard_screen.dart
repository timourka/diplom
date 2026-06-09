import 'package:flutter/material.dart';

import '../api/api_client.dart';

class CommunityLeaderboardScreen extends StatefulWidget {
  const CommunityLeaderboardScreen({super.key});

  @override
  State<CommunityLeaderboardScreen> createState() => _CommunityLeaderboardScreenState();
}

class _CommunityLeaderboardScreenState extends State<CommunityLeaderboardScreen> {
  List<dynamic> _rows = [];
  String? _error;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      _rows = await ApiClient(token: null).topContributors();
    } catch (e) {
      _error = e.toString();
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _medal(int index) {
    switch (index) {
      case 0:
        return '🥇';
      case 1:
        return '🥈';
      case 2:
        return '🥉';
      default:
        return '';
    }
  }

  String _loginOf(dynamic row) {
    if (row is Map) {
      return (row['login'] ?? row['email'] ?? row['userName'] ?? 'Пользователь').toString();
    }
    return 'Пользователь';
  }

  int _approvedCountOf(dynamic row) {
    if (row is Map) {
      final value = row['approvedReportsCount'] ?? row['approvedCount'] ?? row['count'];
      if (value is int) return value;
      if (value is num) return value.toInt();
      return int.tryParse(value?.toString() ?? '') ?? 0;
    }
    return 0;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Вклад сообщества'),
        actions: [
          IconButton(onPressed: _load, icon: const Icon(Icons.refresh)),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(_error!, style: const TextStyle(color: Colors.red)),
                  ),
                )
              : _rows.isEmpty
                  ? const Center(child: Text('Пока нет одобренных отчётов.'))
                  : SingleChildScrollView(
                      padding: const EdgeInsets.all(16),
                      scrollDirection: Axis.horizontal,
                      child: DataTable(
                        columns: const [
                          DataColumn(label: Text('Место')),
                          DataColumn(label: Text('Логин')),
                          DataColumn(label: Text('Одобрено отчётов')),
                        ],
                        rows: List<DataRow>.generate(_rows.length, (index) {
                          final row = _rows[index];
                          final placeText = '${_medal(index)} ${index + 1}'.trim();
                          return DataRow(
                            cells: [
                              DataCell(Text(placeText, style: const TextStyle(fontSize: 20))),
                              DataCell(Text(_loginOf(row))),
                              DataCell(Text(_approvedCountOf(row).toString())),
                            ],
                          );
                        }),
                      ),
                    ),
    );
  }
}
