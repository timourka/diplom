import '../storage/file_key_value_store.dart';

class AuthState {
  static const _tokenKey = 'access_token';
  static const _loginKey = 'last_login';
  String? _token;
  String? _login;

  String? get token => _token;
  String? get login => _login;
  bool get isAuthed => _token != null && _token!.isNotEmpty;

  Future<void> load() async {
    try {
      _token = await FileKeyValueStore.getString(_tokenKey);
      _login = await FileKeyValueStore.getString(_loginKey);
    } catch (_) {
      _token = null;
      _login = null;
    }
  }

  Future<void> setToken(String token, {String? login}) async {
    _token = token;
    if (login != null && login.trim().isNotEmpty) {
      _login = login.trim();
      await FileKeyValueStore.setString(_loginKey, _login!);
    }
    await FileKeyValueStore.setString(_tokenKey, token);
  }

  Future<void> logout() async {
    _token = null;
    _login = null;
    await FileKeyValueStore.remove(_tokenKey);
    await FileKeyValueStore.remove(_loginKey);
  }
}
