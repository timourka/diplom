import '../storage/file_key_value_store.dart';

class AuthState {
  static const _tokenKey = 'access_token';
  String? _token;

  String? get token => _token;
  bool get isAuthed => _token != null && _token!.isNotEmpty;

  Future<void> load() async {
    try {
      _token = await FileKeyValueStore.getString(_tokenKey);
    } catch (_) {
      _token = null;
    }
  }

  Future<void> setToken(String token) async {
    _token = token;
    await FileKeyValueStore.setString(_tokenKey, token);
  }

  Future<void> logout() async {
    _token = null;
    await FileKeyValueStore.remove(_tokenKey);
  }
}
