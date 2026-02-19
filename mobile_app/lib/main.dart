import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import 'auth/auth_state.dart';
import 'screens/scan_screen.dart';

late final List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  cameras = await availableCameras();

  final auth = AuthState();
  await auth.load();

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
