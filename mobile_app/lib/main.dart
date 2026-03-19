import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'api/api_config.dart';
import 'api/model_sync_service.dart';
import 'auth/auth_state.dart';
import 'screens/scan_screen.dart';

late final List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await ApiConfig.load();
  cameras = await availableCameras();

  final auth = AuthState();
  await auth.load();

  final modelSync = ModelSyncService();
  await modelSync.trySyncLatestModel();

  runApp(MyApp(auth: auth));
}

class MyApp extends StatelessWidget {
  final AuthState auth;
  const MyApp({super.key, required this.auth});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ProductsDate',
      theme: ThemeData(useMaterial3: true),
      home: ScanScreen(auth: auth, cameras: cameras),
    );
  }
}
