import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/painting.dart';

import '../ai/detection_result.dart';
import '../ai/tflite_date_detector.dart';
import '../api/model_sync_service.dart';
import '../auth/auth_state.dart';
import '../widgets/manual_add_sheet.dart';
import 'cabinet_screen.dart';
import 'error_dataset_flow_screen.dart';
import 'login_screen.dart';
import 'settings_screen.dart';

class ScanScreen extends StatefulWidget {
  final AuthState auth;
  final List<CameraDescription> cameras;
  final String? startupError;

  const ScanScreen({
    super.key,
    required this.auth,
    required this.cameras,
    this.startupError,
  });

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  CameraController? _controller;
  Future<void>? _initFuture;
  CameraDescription? _selectedCamera;

  final _modelSync = ModelSyncService();
  late final TfliteDateDetector _detector;

  bool _aiReady = false;
  bool _aiBusy = false;
  String _aiStatus = 'Подготовка ИИ...';
  DateTime? _lastInferenceAt;
  Duration? _lastInferenceDuration;
  FramePerf _lastPerf = const FramePerf();
  List<DetectionResult> _detections = const [];
  Map<String, dynamic>? _localModelInfo;
  DateTime _lastInferenceStarted = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    _detector = TfliteDateDetector(modelSyncService: _modelSync);
    if (widget.startupError != null && widget.startupError!.isNotEmpty) {
      _aiStatus = widget.startupError!;
    }
    _initialize();
  }

  Future<void> _refreshAiState() async {
    _localModelInfo = await _modelSync.readLocalModelInfo();
    final ready = await _detector.loadLatestModel();

    if (!mounted) return;

    setState(() {
      _aiReady = ready;
      if (!ready) {
        _aiStatus = 'Локальная модель не найдена. Зайди в настройки и обнови версию ИИ.';
        _detections = const [];
        _lastInferenceAt = null;
      }
    });

    if (ready) {
      await _startImageStream();
    }
  }

  Future<void> _initialize() async {
    if (widget.cameras.isEmpty) {
      if (mounted) {
        setState(() => _aiStatus = 'Камера не найдена или нет разрешения на камеру.');
      }
      return;
    }

    try {
      final back = widget.cameras.where((c) => c.lensDirection == CameraLensDirection.back);
      final cam = back.isNotEmpty ? back.first : widget.cameras.first;
      _selectedCamera = cam;

      final controller = CameraController(
        cam,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      _controller = controller;
      _initFuture = controller.initialize();
      if (mounted) setState(() {});

      await _initFuture;

      _localModelInfo = await _modelSync.readLocalModelInfo();
      final ready = await _detector.loadLatestModel();

      if (!mounted) return;

      setState(() {
        _aiReady = ready;
        _aiStatus = ready
            ? 'ИИ готова.'
            : 'Локальная модель не найдена. Зайди в настройки и обнови версию ИИ.';
      });

      if (ready) {
        await _startImageStream();
      }
    } catch (e) {
      await _controller?.dispose();
      if (!mounted) return;
      setState(() {
        _controller = null;
        _initFuture = null;
        _aiReady = false;
        _aiStatus = 'Ошибка запуска камеры: $e';
      });
    }
  }

  Future<void> _startImageStream() async {
    final controller = _controller;
    if (controller == null || controller.value.isStreamingImages) {
      return;
    }

    await controller.startImageStream((image) async {
      if (!_aiReady || _aiBusy || !mounted) {
        return;
      }

      final now = DateTime.now();
      if (now.difference(_lastInferenceStarted).inMilliseconds < 1000) {
        return;
      }
      _lastInferenceStarted = now;
      _aiBusy = true;
      final startedAt = DateTime.now();

      try {
        final rotationDegrees = _selectedCamera?.sensorOrientation ?? 0;
        final mirror = _selectedCamera?.lensDirection == CameraLensDirection.front;
        final frameResult = await _detector.detectFromCameraImage(
          image,
          rotationDegrees: rotationDegrees,
          mirrorHorizontally: mirror,
        );
        if (!mounted) return;

        final detections = frameResult.detections;
        final perf = frameResult.perf;
        final elapsed = DateTime.now().difference(startedAt);
        final totalDuration = perf.totalMs > 0
            ? Duration(milliseconds: perf.totalMs)
            : elapsed;

        setState(() {
          _detections = detections;
          _lastInferenceAt = DateTime.now();
          _lastInferenceDuration = totalDuration;
          _lastPerf = perf;
          if (detections.isEmpty) {
            _aiStatus = 'Дата на кадре не найдена. Время анализа: ${totalDuration.inMilliseconds} мс.';
          } else {
            final best = detections.first;
            _aiStatus = 'Найдена область даты: ${best.label}, уверенность '
                '${(best.confidence * 100).toStringAsFixed(1)}%. '
                'Время анализа: ${totalDuration.inMilliseconds} мс.';
          }
        });
      } on TimeoutException {
        if (!mounted) return;
        setState(() {
          _lastPerf = const FramePerf();
          _aiStatus = 'Кадр пропущен: ИИ не успела завершить анализ вовремя.';
        });
      } catch (e) {
        if (!mounted) return;
        setState(() {
          _lastPerf = const FramePerf();
          _aiStatus = 'Ошибка ИИ: $e';
        });
      } finally {
        _aiBusy = false;
      }
    });
  }

  @override
  void dispose() {
    unawaited(_controller?.dispose());
    _detector.close();
    super.dispose();
  }

  void _goLogin({required String after}) async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => LoginScreen(auth: widget.auth, after: after)),
    );
    setState(() {});
  }

  void _openSettings() async {
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const SettingsScreen()),
    );
    await _refreshAiState();
  }

  void _onProfile() async {
    if (!widget.auth.isAuthed) return _goLogin(after: 'profile');
    await Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => CabinetScreen(auth: widget.auth)),
    );
    await _refreshAiState();
  }

  void _onManualAdd() async {
    if (!widget.auth.isAuthed) return _goLogin(after: 'manual');
    final added = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      builder: (_) => ManualAddSheet(auth: widget.auth),
    );
    if (added == true && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Продукт добавлен в личный кабинет')),
      );
    }
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

  Widget _buildAiPanel() {
    final version = _localModelInfo?['modelVersionId']?.toString() ?? '—';
    final format = _localModelInfo?['mobileFormat']?.toString() ?? '—';
    final syncedAt = _localModelInfo?['syncedAt']?.toString();
    final lastInferenceText = _lastInferenceAt == null
        ? 'ещё не запускалась'
        : _lastInferenceAt!.toLocal().toString().split('.').first;
    final lastInferenceDuration = _lastInferenceDuration == null
        ? '—'
        : '${_lastInferenceDuration!.inMilliseconds} мс';
    final perfBreakdown = _lastPerf.totalMs == 0 ? '—' : _lastPerf.toPrettyString();

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.65),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Icon(
                _aiReady ? Icons.smart_toy : Icons.smart_toy_outlined,
                color: Colors.white,
              ),
              const SizedBox(width: 8),
              const Expanded(
                child: Text(
                  'ИИ на главном экране',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            _aiStatus,
            style: const TextStyle(color: Colors.white),
          ),
          const SizedBox(height: 10),
          Text(
            'Версия модели: #$version • Формат: $format',
            style: const TextStyle(color: Colors.white70),
          ),
          if (syncedAt != null && syncedAt.isNotEmpty)
            Text(
              'Обновлена: $syncedAt',
              style: const TextStyle(color: Colors.white70),
            ),
          Text(
            'Последний анализ: $lastInferenceText',
            style: const TextStyle(color: Colors.white70),
          ),
          Text(
            'Длительность анализа: $lastInferenceDuration',
            style: const TextStyle(color: Colors.white70),
          ),
          const SizedBox(height: 6),
          Text(
            'Разбивка: $perfBreakdown',
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
          if (_detections.isNotEmpty) ...[
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _detections
                  .take(3)
                  .map(
                    (d) => Chip(
                      label: Text('${d.label} ${(d.confidence * 100).toStringAsFixed(0)}%'),
                      avatar: const Icon(Icons.visibility, size: 18),
                    ),
                  )
                  .toList(),
            ),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: _controller == null
                ? Container(
                    color: Colors.black,
                    alignment: Alignment.center,
                    padding: const EdgeInsets.all(24),
                    child: Text(
                      _aiStatus,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  )
                : FutureBuilder(
                    future: _initFuture,
                    builder: (context, snap) {
                      if (snap.hasError) {
                        return Container(
                          color: Colors.black,
                          alignment: Alignment.center,
                          padding: const EdgeInsets.all(24),
                          child: Text(
                            'Ошибка камеры: ${snap.error}',
                            textAlign: TextAlign.center,
                            style: const TextStyle(color: Colors.white, fontSize: 16),
                          ),
                        );
                      }

                      if (snap.connectionState != ConnectionState.done) {
                        return const Center(child: CircularProgressIndicator());
                      }

                      final previewSize = _controller!.value.previewSize;
                      if (previewSize == null) {
                        return const Center(child: CircularProgressIndicator());
                      }

                      final childSize = Size(previewSize.height, previewSize.width);

                      return Stack(
                        fit: StackFit.expand,
                        children: [
                          FittedBox(
                            fit: BoxFit.cover,
                            child: SizedBox(
                              width: childSize.width,
                              height: childSize.height,
                              child: CameraPreview(_controller!),
                            ),
                          ),
                          IgnorePointer(
                            child: CustomPaint(
                              painter: _DetectionOverlayPainter(
                                detections: _detections,
                                previewContentSize: childSize,
                              ),
                              size: Size.infinite,
                            ),
                          ),
                        ],
                      );
                    },
                  ),
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircleAvatar(
                      radius: 20,
                      child: IconButton(
                        padding: EdgeInsets.zero,
                        onPressed: _openSettings,
                        icon: const Icon(Icons.settings, size: 20),
                        tooltip: 'Настройки API',
                      ),
                    ),
                    const SizedBox(width: 8),
                    GestureDetector(
                      onTap: _onProfile,
                      child: CircleAvatar(
                        radius: 20,
                        child: Icon(widget.auth.isAuthed ? Icons.person : Icons.login),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          SafeArea(
            child: Align(
              alignment: Alignment.topLeft,
              child: _buildAiPanel(),
            ),
          ),
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

class _DetectionOverlayPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final Size previewContentSize;

  const _DetectionOverlayPainter({
    required this.detections,
    required this.previewContentSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty || previewContentSize.isEmpty) {
      return;
    }

    final fittedSizes = applyBoxFit(BoxFit.cover, previewContentSize, size);
    final source = fittedSizes.source;
    final destination = fittedSizes.destination;
    final dx = (size.width - destination.width) / 2;
    final dy = (size.height - destination.height) / 2;
    final dstRect = Offset(dx, dy) & destination;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = const Color(0xFF6BEE8A);

    final fill = Paint()..color = const Color(0xCC1C1C1C);

    for (final detection in detections) {
      final left = dstRect.left + detection.left * dstRect.width;
      final top = dstRect.top + detection.top * dstRect.height;
      final right = dstRect.left + detection.right * dstRect.width;
      final bottom = dstRect.top + detection.bottom * dstRect.height;

      final rect = Rect.fromLTRB(
        left.clamp(dstRect.left, dstRect.right),
        top.clamp(dstRect.top, dstRect.bottom),
        right.clamp(dstRect.left, dstRect.right),
        bottom.clamp(dstRect.top, dstRect.bottom),
      );

      if (rect.width < 4 || rect.height < 4) {
        continue;
      }

      canvas.drawRect(rect, paint);

      final label = '${detection.label} ${(detection.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: math.max(80, rect.width));

      final bubble = RRect.fromRectAndRadius(
        Rect.fromLTWH(
          rect.left,
          math.max(dstRect.top, rect.top - textPainter.height - 10),
          textPainter.width + 14,
          textPainter.height + 8,
        ),
        const Radius.circular(8),
      );

      canvas.drawRRect(bubble, fill);
      textPainter.paint(canvas, Offset(bubble.left + 7, bubble.top + 4));
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionOverlayPainter oldDelegate) {
    return oldDelegate.detections != detections || oldDelegate.previewContentSize != previewContentSize;
  }
}
