import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import 'api_config.dart';

class AuthRequiredException implements Exception {
  final String message;
  const AuthRequiredException([this.message = 'Нужно заново войти в аккаунт']);
  @override
  String toString() => message;
}

class NetworkApiException implements Exception {
  final String message;
  const NetworkApiException([this.message = 'Нет соединения с сервером']);
  @override
  String toString() => message;
}

class ApiRequestException implements Exception {
  final int statusCode;
  final String body;
  const ApiRequestException(this.statusCode, this.body);
  @override
  String toString() => body.isEmpty ? 'HTTP $statusCode' : body;
}

class ApiClient {
  final String? token;
  ApiClient({required this.token});

  static String get baseUrl => ApiConfig.baseUrl;

  Map<String, String> _headers({bool auth = false}) {
    final h = {'Content-Type': 'application/json'};
    if (auth && token != null && token!.isNotEmpty) {
      h['Authorization'] = 'Bearer $token';
    }
    return h;
  }

  Future<http.Response> _safeRequest(Future<http.Response> Function() action) async {
    try {
      return await action().timeout(const Duration(seconds: 20));
    } on AuthRequiredException {
      rethrow;
    } on SocketException {
      throw const NetworkApiException();
    } on TimeoutException {
      throw const NetworkApiException('Сервер долго не отвечает');
    } on http.ClientException catch (e) {
      throw NetworkApiException('Не удалось подключиться к серверу: ${e.message}');
    }
  }

  Future<String> _readOkToken(http.Response r) async {
    _throwIfBad(r);
    final decoded = jsonDecode(r.body);
    return decoded['accessToken']?.toString() ?? '';
  }

  void _throwIfBad(http.Response r) {
    if (r.statusCode == 401 || r.statusCode == 403) {
      throw const AuthRequiredException();
    }
    if (r.statusCode < 200 || r.statusCode >= 300) {
      throw ApiRequestException(r.statusCode, r.body);
    }
  }

  Future<String> register(String login, String password) async {
    final r = await _safeRequest(
      () => http.post(
        Uri.parse('$baseUrl/api/Auth/register'),
        headers: _headers(),
        body: jsonEncode({'email': login, 'password': password}),
      ),
    );
    return _readOkToken(r);
  }

  Future<String> login(String login, String password) async {
    final r = await _safeRequest(
      () => http.post(
        Uri.parse('$baseUrl/api/Auth/login'),
        headers: _headers(),
        body: jsonEncode({'email': login, 'password': password}),
      ),
    );
    return _readOkToken(r);
  }

  Future<Map<String, dynamic>> me() async {
    final r = await _safeRequest(
      () => http.get(
        Uri.parse('$baseUrl/api/Users/me'),
        headers: _headers(auth: true),
      ),
    );
    _throwIfBad(r);
    final decoded = jsonDecode(r.body);
    return decoded is Map<String, dynamic> ? decoded : <String, dynamic>{};
  }

  Future<List<dynamic>> products() async {
    final r = await _safeRequest(
      () => http.get(
        Uri.parse('$baseUrl/api/Products'),
        headers: _headers(),
      ),
    );
    _throwIfBad(r);
    final decoded = jsonDecode(r.body);
    return decoded is List ? decoded : <dynamic>[];
  }

  Future<List<dynamic>> storage() async {
    final r = await _safeRequest(
      () => http.get(
        Uri.parse('$baseUrl/api/Storage'),
        headers: _headers(auth: true),
      ),
    );
    _throwIfBad(r);
    final decoded = jsonDecode(r.body);
    return decoded is List ? decoded : <dynamic>[];
  }

  Future<Map<String, dynamic>> addStoredProductByName(String productName, DateTime expiryAt) async {
    final r = await _safeRequest(
      () => http.post(
        Uri.parse('$baseUrl/api/Storage'),
        headers: _headers(auth: true),
        body: jsonEncode({
          'productName': productName,
          'manufactureAt': null,
          'expiryAt': expiryAt.toIso8601String(),
        }),
      ),
    );
    _throwIfBad(r);
    final decoded = jsonDecode(r.body);
    return decoded is Map<String, dynamic> ? decoded : <String, dynamic>{};
  }

  Future<void> deleteStoredProduct(int id) async {
    final r = await _safeRequest(
      () => http.delete(
        Uri.parse('$baseUrl/api/Storage/$id'),
        headers: _headers(auth: true),
      ),
    );
    _throwIfBad(r);
  }

  Future<List<dynamic>> topContributors() async {
    final r = await _safeRequest(
      () => http.get(
        Uri.parse('$baseUrl/api/community/top-contributors'),
        headers: _headers(),
      ),
    );
    _throwIfBad(r);
    final decoded = jsonDecode(r.body);
    return decoded is List ? decoded : <dynamic>[];
  }

  Future<void> uploadDatasetZip(String zipPath, {String comment = ''}) async {
    final uri = Uri.parse('$baseUrl/api/error-reports/upload-dataset');
    final req = http.MultipartRequest('POST', uri);

    if (token != null && token!.isNotEmpty) {
      req.headers['Authorization'] = 'Bearer $token';
    }
    req.fields['comment'] = comment;
    req.files.add(await http.MultipartFile.fromPath('datasetZip', zipPath));

    http.StreamedResponse res;
    try {
      res = await req.send().timeout(const Duration(seconds: 60));
    } on SocketException {
      throw const NetworkApiException();
    } on TimeoutException {
      throw const NetworkApiException('Сервер долго не отвечает');
    } on http.ClientException catch (e) {
      throw NetworkApiException('Не удалось подключиться к серверу: ${e.message}');
    }

    final body = await res.stream.bytesToString();
    if (res.statusCode == 401 || res.statusCode == 403) {
      throw const AuthRequiredException();
    }
    if (res.statusCode < 200 || res.statusCode >= 300) {
      throw ApiRequestException(res.statusCode, body);
    }
  }
}
