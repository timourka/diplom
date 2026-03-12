import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../auth/auth_state.dart';
import '../widgets/manual_add_sheet.dart';
import 'error_dataset_flow_screen.dart';
import 'login_screen.dart';
import 'cabinet_screen.dart';

class ScanScreen extends StatefulWidget {
  final AuthState auth;
  final List<CameraDescription> cameras;

  const ScanScreen({super.key, required this.auth, required this.cameras});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  CameraController? _controller;
  Future<void>? _initFuture;

  @override
  void initState() {
    super.initState();

    // Берём заднюю камеру, если есть, иначе первую
    final back = widget.cameras.where((c) => c.lensDirection == CameraLensDirection.back);
    final cam = back.isNotEmpty ? back.first : widget.cameras.first;

    _controller = CameraController(
      cam,
      ResolutionPreset.high,
      enableAudio: false,
    );

    _initFuture = _controller!.initialize();
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  void _goLogin({required String after}) async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => LoginScreen(auth: widget.auth, after: after)),
    );
    setState(() {});
  }

  void _onProfile() {
    if (!widget.auth.isAuthed) return _goLogin(after: 'profile');
    Navigator.push(context, MaterialPageRoute(builder: (_) => CabinetScreen(auth: widget.auth)));
  }

  void _onManualAdd() async {
    if (!widget.auth.isAuthed) return _goLogin(after: 'manual');
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => ManualAddSheet(auth: widget.auth),
    );
  }

  void _onError() {
    if (!widget.auth.isAuthed) return _goLogin(after: 'error');
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ErrorDatasetFlowScreen(auth: widget.auth, cameras: widget.cameras),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Реальный preview камеры
          Positioned.fill(
            child: _controller == null
                ? const Center(child: CircularProgressIndicator())
                : FutureBuilder(
              future: _initFuture,
              builder: (context, snap) {
                if (snap.connectionState != ConnectionState.done) {
                  return const Center(child: CircularProgressIndicator());
                }
                return FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: _controller!.value.previewSize!.height,
                    height: _controller!.value.previewSize!.width,
                    child: CameraPreview(_controller!),
                  ),
                );
              },
            ),
          ),

          // Профиль справа сверху
          SafeArea(
            child: Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: GestureDetector(
                  onTap: _onProfile,
                  child: CircleAvatar(
                    radius: 20,
                    child: Icon(widget.auth.isAuthed ? Icons.person : Icons.login),
                  ),
                ),
              ),
            ),
          ),

          // Кнопки снизу
          SafeArea(
            child: Align(
              alignment: Alignment.bottomLeft,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: FloatingActionButton(
                  heroTag: 'manualAdd',
                  onPressed: _onManualAdd,
                  child: const Icon(Icons.add),
                ),
              ),
            ),
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.bottomRight,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: FloatingActionButton.extended(
                  heroTag: 'error',
                  onPressed: _onError,
                  icon: const Icon(Icons.report),
                  label: const Text('Сообщить'),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
