import 'package:shared_preferences/shared_preferences.dart';

class AuthState {
  static const _tokenKey = 'access_token';
  String? _token;

  String? get token => _token;
  bool get isAuthed => _token != null && _token!.isNotEmpty;

  Future<void> load() async {
    final sp = await SharedPreferences.getInstance();
    _token = sp.getString(_tokenKey);
  }

  Future<void> setToken(String token) async {
    _token = token;
    final sp = await SharedPreferences.getInstance();
    await sp.setString(_tokenKey, token);
  }

  Future<void> logout() async {
    _token = null;
    final sp = await SharedPreferences.getInstance();
    await sp.remove(_tokenKey);
  }
}
