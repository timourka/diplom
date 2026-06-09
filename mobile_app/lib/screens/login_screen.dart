import 'package:flutter/material.dart';
import '../auth/auth_state.dart';
import '../api/api_client.dart';
import '../api/api_config.dart';
import 'register_screen.dart';
import 'settings_screen.dart';

class LoginScreen extends StatefulWidget {
  final AuthState auth;
  final String after;
  final String title;
  final String? message;
  const LoginScreen({
    super.key,
    required this.auth,
    required this.after,
    this.title = 'Авторизация',
    this.message,
  });

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _email = TextEditingController();
  final _pass = TextEditingController();

  String? _error;
  bool _loading = false;

  Future<void> _openSettings() async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const SettingsScreen()),
    );
    if (mounted) setState(() {});
  }

  Future<void> _login() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final api = ApiClient(token: null);
      final login = _email.text.trim();
      final token = await api.login(login, _pass.text);
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
    _email.dispose();
    _pass.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
          IconButton(
            onPressed: _openSettings,
            icon: const Icon(Icons.settings),
            tooltip: 'Настройки API',
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _email,
                decoration: const InputDecoration(labelText: 'Логин'),
                keyboardType: TextInputType.text,
              ),
              TextField(
                controller: _pass,
                decoration: const InputDecoration(labelText: 'Пароль'),
                obscureText: true,
              ),
              const SizedBox(height: 12),
              Card(
                child: ListTile(
                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                  leading: const Icon(Icons.lan),
                  title: const Text('Адрес API'),
                  subtitle: Text(ApiConfig.baseUrl),
                  trailing: TextButton(
                    onPressed: _loading ? null : _openSettings,
                    child: const Text('Изменить'),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              if (_error != null) Text(_error!, style: const TextStyle(color: Colors.red)),
              const SizedBox(height: 12),
              ElevatedButton(
                onPressed: _loading ? null : _login,
                child: Text(_loading ? 'Вход...' : 'Войти'),
              ),
              TextButton(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => RegisterScreen(auth: widget.auth)),
                ),
                child: const Text('Нет аккаунта? Регистрация'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
