import 'dart:async';
import 'dart:developer' as developer;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'api/api_config.dart';
import 'api/model_sync_service.dart';
import 'auth/auth_state.dart';
import 'screens/scan_screen.dart';

List<CameraDescription> cameras = const [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  String? startupError;

  try {
    await ApiConfig.load();
  } catch (e, st) {
    startupError = 'Не удалось загрузить настройки API: $e';
    developer.log(startupError, name: 'Startup', error: e, stackTrace: st);
  }

  try {
    cameras = await availableCameras();
  } catch (e, st) {
    startupError = 'Не удалось получить доступ к камере: $e';
    developer.log(startupError, name: 'Startup', error: e, stackTrace: st);
  }

  final auth = AuthState();
  try {
    await auth.load();
  } catch (e, st) {
    developer.log('Не удалось загрузить авторизацию: $e', name: 'Startup', error: e, stackTrace: st);
  }

  // Не блокируем запуск приложения синхронизацией модели.
  // Если сервер недоступен или модель ещё не опубликована, пользователь всё равно увидит интерфейс.
  unawaited(ModelSyncService().trySyncLatestModel());

  runApp(MyApp(auth: auth, startupError: startupError));
}

class MyApp extends StatelessWidget {
  final AuthState auth;
  final String? startupError;

  const MyApp({super.key, required this.auth, this.startupError});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ProductsDate',
      theme: ThemeData(useMaterial3: true),
      home: ScanScreen(auth: auth, cameras: cameras, startupError: startupError),
    );
  }
}
