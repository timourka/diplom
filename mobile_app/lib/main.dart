import 'dart:async';
import 'dart:developer' as developer;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'api/api_config.dart';
import 'api/model_sync_service.dart';
import 'auth/auth_state.dart';
import 'models/stored_product.dart';
import 'services/expiry_notification_service.dart';
import 'services/local_storage_repository.dart';
import 'services/offline_sync_service.dart';
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

  try {
    final localStorage = LocalStoredProductRepository();
    final cached = await localStorage.mergeCachedWithPending(await localStorage.readCachedStorage());
    final items = cached
        .whereType<Map>()
        .map((x) => StoredProduct.fromJson(x.map((key, value) => MapEntry(key.toString(), value))))
        .toList(growable: false);
    await ExpiryNotificationService().notifyDueToday(items);
    unawaited(OfflineSyncService().trySync(auth));
  } catch (e, st) {
    developer.log('Не удалось выполнить локальные уведомления/синхронизацию: $e', name: 'Startup', error: e, stackTrace: st);
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
      locale: const Locale('ru', 'RU'),
      supportedLocales: const [
        Locale('ru', 'RU'),
        Locale('en', 'US'),
      ],
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      theme: ThemeData(useMaterial3: true),
      home: ScanScreen(auth: auth, cameras: cameras, startupError: startupError),
    );
  }
}
