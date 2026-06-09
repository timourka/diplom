import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../screens/login_screen.dart';
import 'auth_state.dart';

class AuthFlow {
  static Future<T> runWithReauth<T>({
    required BuildContext context,
    required AuthState auth,
    required Future<T> Function() action,
    String after = 'reauth',
  }) async {
    try {
      return await action();
    } on AuthRequiredException {
      if (!context.mounted) rethrow;
      final loggedIn = await Navigator.push<bool>(
        context,
        MaterialPageRoute(
          builder: (_) => LoginScreen(
            auth: auth,
            after: after,
            title: 'Авторизация',
          ),
        ),
      );
      if (loggedIn == true) {
        return await action();
      }
      throw const AuthRequiredException('Авторизация отменена');
    }
  }
}
