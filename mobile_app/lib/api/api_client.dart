import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiClient {
  /// Android emulator -> host machine:
  /// http://10.0.2.2:<port>
  static const baseUrl = 'http://10.0.2.2:5099';

  final String? token;
  ApiClient({required this.token});

  Map<String, String> _headers({bool auth = false}) {
    final h = {'Content-Type': 'application/json'};
    if (auth && token != null && token!.isNotEmpty) {
      h['Authorization'] = 'Bearer $token';
    }
    return h;
  }

  Future<String> register(String email, String password) async {
    final r = await http.post(
      Uri.parse('$baseUrl/api/Auth/register'),
      headers: _headers(),
      body: jsonEncode({'email': email, 'password': password}),
    );
    if (r.statusCode != 200) throw Exception(r.body);
    return (jsonDecode(r.body) as Map<String, dynamic>)['accessToken'] as String;
  }

  Future<String> login(String email, String password) async {
    final r = await http.post(
      Uri.parse('$baseUrl/api/Auth/login'),
      headers: _headers(),
      body: jsonEncode({'email': email, 'password': password}),
    );
    if (r.statusCode != 200) throw Exception(r.body);
    return (jsonDecode(r.body) as Map<String, dynamic>)['accessToken'] as String;
  }

  Future<Map<String, dynamic>> me() async {
    final r = await http.get(
      Uri.parse('$baseUrl/api/Users/me'),
      headers: _headers(auth: true),
    );
    if (r.statusCode != 200) throw Exception(r.body);
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<List<dynamic>> products() async {
    final r = await http.get(
      Uri.parse('$baseUrl/api/Products'),
      headers: _headers(),
    );
    if (r.statusCode != 200) throw Exception(r.body);
    return jsonDecode(r.body) as List<dynamic>;
  }

  Future<List<dynamic>> storage() async {
    final r = await http.get(
      Uri.parse('$baseUrl/api/Storage'),
      headers: _headers(auth: true),
    );
    if (r.statusCode != 200) throw Exception(r.body);
    return jsonDecode(r.body) as List<dynamic>;
  }

  Future<void> addStoredProduct(int productId, DateTime? expiryAt) async {
    final r = await http.post(
      Uri.parse('$baseUrl/api/Storage'),
      headers: _headers(auth: true),
      body: jsonEncode({
        'productId': productId,
        'manufactureAt': null,
        'expiryAt': expiryAt?.toIso8601String(),
      }),
    );
    if (r.statusCode != 200) throw Exception(r.body);
  }
}
