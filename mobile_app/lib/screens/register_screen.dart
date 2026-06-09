import 'package:flutter/material.dart';
import '../auth/auth_state.dart';
import '../api/api_client.dart';

class RegisterScreen extends StatefulWidget {
  final AuthState auth;
  const RegisterScreen({super.key, required this.auth});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _login = TextEditingController();
  final _pass = TextEditingController();

  String? _error;
  bool _loading = false;

  Future<void> _register() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final api = ApiClient(token: null);
      final login = _login.text.trim();
      final token = await api.register(login, _pass.text);
      await widget.auth.setToken(token, login: login);

      if (mounted) Navigator.pop(context, true);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _login.dispose();
    _pass.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Регистрация')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: _login,
              decoration: const InputDecoration(labelText: 'Логин'),
              keyboardType: TextInputType.text,
            ),
            TextField(
              controller: _pass,
              decoration: const InputDecoration(labelText: 'Пароль'),
              obscureText: true,
            ),
            const SizedBox(height: 12),
            if (_error != null) Text(_error!, style: const TextStyle(color: Colors.red)),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: _loading ? null : _register,
              child: Text(_loading ? 'Создание...' : 'Создать аккаунт'),
            ),
          ],
        ),
      ),
    );
  }
}
