import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/painting.dart';
import 'package:image/image.dart' as img;

import '../ai/date_ocr_service.dart';
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

class _ScanScreenState extends State<ScanScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initFuture;
  CameraDescription? _selectedCamera;

  final _modelSync = ModelSyncService();
  late final TfliteDateDetector _detector;
  late final DateOcrService _dateOcr;

  bool _aiReady = false;
  bool _aiBusy = false;
  bool _scanActive = false;
  bool _coveredByChildRoute = false;
  bool _initializingCamera = false;
  bool _resumeRequestedAfterInit = false;
  int _cameraEpoch = 0;
  int _previewKeySeed = 0;
  String _aiStatus = 'Подготовка ИИ...';
  DateTime? _lastInferenceAt;
  Duration? _lastInferenceDuration;
  FramePerf _lastPerf = const FramePerf();
  List<DetectionResult> _detections = const [];
  bool _magnifierBusy = false;
  int _magnifierEpoch = 0;
  File? _magnifierImageFile;
  String _magnifierStatus = '';
  DetectionResult? _magnifierSourceDetection;
  DateTime? _magnifierRecognizedDate;
  String? _magnifierRecognizedDateText;
  bool _manualAddSheetOpen = false;
  Map<String, dynamic>? _localModelInfo;
  DateTime _lastInferenceStarted = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _detector = TfliteDateDetector(modelSyncService: _modelSync);
    _dateOcr = DateOcrService();
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

  Future<void> _initialize({bool forceRecreate = false}) async {
    if (_initializingCamera) {
      _resumeRequestedAfterInit = true;
      return;
    }

    _initializingCamera = true;

    try {
      if (forceRecreate) {
        await _disposeCameraController(updateState: false);
      }

      if (widget.cameras.isEmpty) {
        if (mounted) {
          setState(() => _aiStatus = 'Камера не найдена или нет разрешения на камеру.');
        }
        return;
      }

      if (_coveredByChildRoute) {
        return;
      }

      if (!forceRecreate && _controller != null && _controller!.value.isInitialized) {
        await _refreshAiState();
        return;
      }

      final back = widget.cameras.where((c) => c.lensDirection == CameraLensDirection.back);
      final cam = back.isNotEmpty ? back.first : widget.cameras.first;
      _selectedCamera = cam;

      final localEpoch = ++_cameraEpoch;
      final controller = CameraController(
        cam,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      _controller = controller;
      _initFuture = controller.initialize();
      _previewKeySeed++;
      if (mounted) {
        setState(() {
          _detections = const [];
          _lastInferenceAt = null;
          _aiStatus = 'Запускаю камеру...';
        });
      }

      await _initFuture;

      if (!mounted || _coveredByChildRoute || localEpoch != _cameraEpoch) {
        try {
          await controller.dispose();
        } catch (_) {
          // Controller was superseded or the route is covered; native camera will be released by dispose.
        }
        return;
      }

      _localModelInfo = await _modelSync.readLocalModelInfo();
      final ready = await _detector.loadLatestModel();

      if (!mounted || _coveredByChildRoute || localEpoch != _cameraEpoch) return;

      setState(() {
        _aiReady = ready;
        _aiStatus = ready
            ? 'ИИ готова.'
            : 'Локальная модель не найдена. Зайди в настройки и обнови версию ИИ.';
      });

      if (ready) {
        await _startImageStream(expectedEpoch: localEpoch);
      }
    } catch (e) {
      await _disposeCameraController(updateState: false);
      if (!mounted) return;
      setState(() {
        _controller = null;
        _initFuture = null;
        _aiReady = false;
        _scanActive = false;
        _aiStatus = 'Ошибка запуска камеры: $e';
      });
    } finally {
      _initializingCamera = false;
      if (_resumeRequestedAfterInit && mounted && !_coveredByChildRoute) {
        _resumeRequestedAfterInit = false;
        unawaited(_resumeScanning(recreateCamera: true));
      } else {
        _resumeRequestedAfterInit = false;
      }
    }
  }

  Future<void> _stopImageStreamIfNeeded() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || !controller.value.isStreamingImages) {
      return;
    }

    try {
      await controller.stopImageStream();
    } catch (_) {
      // The camera plugin may throw if the stream is already stopping/disposed.
    }
  }

  Future<void> _disposeCameraController({bool updateState = true}) async {
    _scanActive = false;
    _aiBusy = false;
    _cameraEpoch++;

    final controller = _controller;
    _controller = null;
    _initFuture = null;
    _previewKeySeed++;

    if (updateState && mounted) {
      setState(() {
        _detections = const [];
        _lastInferenceAt = null;
      });
    }

    if (controller == null) return;

    try {
      if (controller.value.isInitialized && controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (_) {
      // Ignore stop errors during disposal; disposing releases the native camera.
    }

    try {
      await controller.dispose();
    } catch (_) {
      // Ignore double-dispose/native disposal races.
    }
  }

  Future<void> _pauseScanning({bool releaseCamera = false}) async {
    _scanActive = false;
    _aiBusy = false;
    _cameraEpoch++;

    if (mounted) {
      setState(() {
        _detections = const [];
        _aiStatus = releaseCamera ? 'Камера освобождена для другого экрана.' : 'Сканирование приостановлено.';
      });
    }

    if (releaseCamera) {
      await _disposeCameraController(updateState: true);
    } else {
      await _stopImageStreamIfNeeded();
    }
  }

  Future<void> _resumeScanning({required bool recreateCamera}) async {
    if (!mounted || _coveredByChildRoute || _magnifierActive) return;

    // Дожидаемся первого кадра после закрытия route, чтобы CameraPreview получил новый Surface,
    // а предыдущий экран успел освободить CameraX use cases до инициализации новой камеры.
    await WidgetsBinding.instance.endOfFrame;
    if (!mounted || _coveredByChildRoute) return;

    if (recreateCamera || _controller == null || !_controller!.value.isInitialized) {
      await _initialize(forceRecreate: recreateCamera);
    } else {
      await _refreshAiState();
    }
  }

  Future<T?> _pushWithScanPaused<T>(Route<T> route, {bool releaseCamera = false}) async {
    _coveredByChildRoute = true;
    await _pauseScanning(releaseCamera: releaseCamera);

    try {
      return await Navigator.push(context, route);
    } finally {
      _coveredByChildRoute = false;
      await _resumeScanning(recreateCamera: releaseCamera);
    }
  }

  Future<T?> _showModalWithScanPaused<T>(Future<T?> Function() show) async {
    _coveredByChildRoute = true;
    await _pauseScanning(releaseCamera: false);

    try {
      return await show();
    } finally {
      _coveredByChildRoute = false;
      await _resumeScanning(recreateCamera: false);
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused ||
        state == AppLifecycleState.detached ||
        state == AppLifecycleState.hidden) {
      unawaited(_pauseScanning(releaseCamera: true));
      return;
    }

    if (state == AppLifecycleState.resumed && !_coveredByChildRoute && !_magnifierActive) {
      unawaited(_resumeScanning(recreateCamera: true));
    }
  }

  Future<void> _startImageStream({int? expectedEpoch}) async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _coveredByChildRoute) {
      _scanActive = false;
      return;
    }

    if (controller.value.isStreamingImages) {
      if (_scanActive) return;
      try {
        await controller.stopImageStream();
      } catch (_) {
        // If plugin reports stale streaming state, recreate the camera cleanly.
        await _disposeCameraController(updateState: true);
        if (mounted && !_coveredByChildRoute) {
          await _initialize(forceRecreate: true);
        }
        return;
      }
    }

    final streamEpoch = expectedEpoch ?? _cameraEpoch;
    _scanActive = true;

    await controller.startImageStream((image) async {
      if (!_scanActive || _cameraEpoch != streamEpoch || !_aiReady || _aiBusy || !mounted) {
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
        if (!mounted || !_scanActive || _cameraEpoch != streamEpoch) return;

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
        if (!mounted || !_scanActive || _cameraEpoch != streamEpoch) return;
        setState(() {
          _lastPerf = const FramePerf();
          _aiStatus = 'Кадр пропущен: ИИ не успела завершить анализ вовремя.';
        });
      } catch (e) {
        if (!mounted || !_scanActive || _cameraEpoch != streamEpoch) return;
        setState(() {
          _lastPerf = const FramePerf();
          _aiStatus = 'Ошибка ИИ: $e';
        });
      } finally {
        if (_cameraEpoch == streamEpoch) {
          _aiBusy = false;
        }
      }
    });
  }


  bool get _magnifierActive => _magnifierBusy || _magnifierImageFile != null;

  Rect _displayRectForDetection(DetectionResult detection, Size viewportSize, Size previewContentSize) {
    if (previewContentSize.isEmpty || viewportSize.isEmpty) {
      return Rect.zero;
    }

    final fittedSizes = applyBoxFit(BoxFit.cover, previewContentSize, viewportSize);
    final destination = fittedSizes.destination;
    final dx = (viewportSize.width - destination.width) / 2;
    final dy = (viewportSize.height - destination.height) / 2;
    final dstRect = Offset(dx, dy) & destination;

    return Rect.fromLTRB(
      dstRect.left + detection.left * dstRect.width,
      dstRect.top + detection.top * dstRect.height,
      dstRect.left + detection.right * dstRect.width,
      dstRect.top + detection.bottom * dstRect.height,
    );
  }

  DetectionResult? _hitTestDetection(Offset position, Size viewportSize, Size previewContentSize) {
    if (_detections.isEmpty) return null;

    final hits = <({DetectionResult detection, Rect rect})>[];
    for (final detection in _detections) {
      final rect = _displayRectForDetection(detection, viewportSize, previewContentSize).inflate(14);
      if (rect.contains(position)) {
        hits.add((detection: detection, rect: rect));
      }
    }

    if (hits.isEmpty) return null;
    hits.sort((a, b) => (a.rect.width * a.rect.height).compareTo(b.rect.width * b.rect.height));
    return hits.first.detection;
  }

  Future<void> _onDetectionTap(DetectionResult detection) async {
    if (_magnifierBusy || !mounted) return;

    final magnifierEpoch = ++_magnifierEpoch;

    setState(() {
      _magnifierBusy = true;
      _magnifierStatus = 'Делаю снимок высокого разрешения и уточняю границы даты...';
      _magnifierSourceDetection = detection;
      _magnifierImageFile = null;
      _magnifierRecognizedDate = null;
      _magnifierRecognizedDateText = null;
    });

    XFile? highResShot;
    File? cropFile;

    try {
      await _pauseScanning(releaseCamera: true);
      // Даём CameraX на Android немного времени закрыть предыдущую сессию перед новым high-res capture.
      await Future<void>.delayed(const Duration(milliseconds: 250));

      highResShot = await _captureHighResolutionStill(focusDetection: detection);

      if (!mounted || magnifierEpoch != _magnifierEpoch) {
        try {
          await File(highResShot.path).delete();
        } catch (_) {}
        return;
      }
      setState(() {
        _magnifierStatus = 'Уточняю границы даты на снимке высокого разрешения...';
      });

      final refined = await _detector.detectFromImageFile(
        highResShot.path,
        focusRegion: detection,
        focusPadding: 1.6,
        confidenceThreshold: 0.01,
        maxDetections: 30,
        strictGeometryFilters: false,
      );

      final refinedDetection = _pickBestRefinedDetection(refined.detections, detection);
      final hasRefinedDetection = refinedDetection != null;
      cropFile = await _createMagnifierCrop(
        imagePath: highResShot.path,
        detection: refinedDetection ?? detection,
        fallbackDetection: detection,
        refined: hasRefinedDetection,
      );

      try {
        await File(highResShot.path).delete();
      } catch (_) {
        // Temporary camera file cleanup is best-effort.
      }

      DateOcrResult? ocrResult;
      try {
        if (cropFile != null) {
          ocrResult = await _dateOcr.recognizeDateFromImagePath(cropFile.path);
        }
      } catch (_) {
        ocrResult = null;
      }

      if (!mounted || magnifierEpoch != _magnifierEpoch) {
        try {
          if (cropFile != null && await cropFile.exists()) await cropFile.delete();
        } catch (_) {}
        return;
      }

      final recognizedDate = ocrResult?.date;
      final recognizedText = ocrResult?.matchedText;
      final baseStatus = hasRefinedDetection
          ? 'Область уточнена на снимке высокого разрешения. Уверенность: '
              '${((refinedDetection.confidence) * 100).toStringAsFixed(1)}%.'
          : 'Показан фрагмент по месту клика.';

      setState(() {
        _magnifierBusy = false;
        _magnifierImageFile = cropFile;
        _magnifierRecognizedDate = recognizedDate;
        _magnifierRecognizedDateText = recognizedText;
        _magnifierStatus = recognizedDate == null
            ? '$baseStatus Дата не распознана.'
            : '$baseStatus Распознана дата: ${_formatDateRu(recognizedDate)}.';
      });

      if (recognizedDate != null) {
        await Future<void>.delayed(const Duration(milliseconds: 150));
        if (mounted && magnifierEpoch == _magnifierEpoch && _magnifierImageFile != null) {
          unawaited(_openManualAdd(initialExpiry: recognizedDate, keepMagnifier: true));
        }
      } else {
        _showSnackBar('Дата не распознана');
      }
    } catch (e) {
      try {
        if (highResShot != null) await File(highResShot.path).delete();
      } catch (_) {}
      try {
        if (cropFile != null && await cropFile.exists()) await cropFile.delete();
      } catch (_) {}

      if (!mounted || magnifierEpoch != _magnifierEpoch) {
        try {
          if (cropFile != null && await cropFile.exists()) await cropFile.delete();
        } catch (_) {}
        return;
      }
      setState(() {
        _magnifierBusy = false;
        _magnifierImageFile = null;
        _magnifierSourceDetection = null;
        _magnifierRecognizedDate = null;
        _magnifierRecognizedDateText = null;
        _magnifierStatus = '';
        _aiStatus = 'Не удалось открыть лупу: $e';
      });
      await _resumeScanning(recreateCamera: true);
    }
  }

  DetectionResult? _pickBestRefinedDetection(
    List<DetectionResult> candidates,
    DetectionResult source,
  ) {
    if (candidates.isEmpty) return null;

    final sourceCenterX = (source.left + source.right) / 2.0;
    final sourceCenterY = (source.top + source.bottom) / 2.0;
    final sourceW = math.max(0.001, source.right - source.left);
    final sourceH = math.max(0.001, source.bottom - source.top);
    final expandedSource = Rect.fromLTRB(
      (source.left - sourceW * 1.2).clamp(0.0, 1.0).toDouble(),
      (source.top - sourceH * 2.0).clamp(0.0, 1.0).toDouble(),
      (source.right + sourceW * 1.2).clamp(0.0, 1.0).toDouble(),
      (source.bottom + sourceH * 2.0).clamp(0.0, 1.0).toDouble(),
    );

    final scored = candidates.map((candidate) {
      final centerX = (candidate.left + candidate.right) / 2.0;
      final centerY = (candidate.top + candidate.bottom) / 2.0;
      final dx = (centerX - sourceCenterX) / math.max(0.05, sourceW * 2.0);
      final dy = (centerY - sourceCenterY) / math.max(0.05, sourceH * 4.0);
      final distance = math.sqrt(dx * dx + dy * dy);
      final candidateCenter = Offset(centerX, centerY);
      final insideBonus = expandedSource.contains(candidateCenter) ? 0.35 : 0.0;
      final score = candidate.confidence + insideBonus - distance * 0.22;
      return (candidate: candidate, score: score);
    }).toList(growable: false)
      ..sort((a, b) => b.score.compareTo(a.score));

    final best = scored.first;
    // Если лучший кандидат совсем далеко от точки клика, лучше показать исходный фрагмент,
    // чем уверенно обрезать чужой текст на упаковке.
    if (best.score < -0.35) return null;
    return best.candidate;
  }

  Future<XFile> _captureHighResolutionStill({DetectionResult? focusDetection}) async {
    if (widget.cameras.isEmpty) {
      throw StateError('Камера не найдена.');
    }

    final cam = _selectedCamera ??
        widget.cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => widget.cameras.first,
        );

    final highResController = CameraController(
      cam,
      ResolutionPreset.max,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await highResController.initialize();
      try {
        await highResController.setFlashMode(FlashMode.off);
      } catch (_) {
        // Not all devices support explicit flash configuration for this temporary controller.
      }

      final focus = focusDetection;
      if (focus != null) {
        final focusPoint = Offset(
          ((focus.left + focus.right) / 2.0).clamp(0.0, 1.0).toDouble(),
          ((focus.top + focus.bottom) / 2.0).clamp(0.0, 1.0).toDouble(),
        );

        try {
          await highResController.setFocusMode(FocusMode.auto);
        } catch (_) {}
        try {
          await highResController.setExposureMode(ExposureMode.auto);
        } catch (_) {}
        try {
          await highResController.setFocusPoint(focusPoint);
        } catch (_) {}
        try {
          await highResController.setExposurePoint(focusPoint);
        } catch (_) {}

        // Flutter camera не отдаёт событие "фокус навёлся". Поэтому не ждём фиксированную секунду:
        // даём короткую паузу только на применение focus/exposure point и сразу снимаем.
        await Future<void>.delayed(const Duration(milliseconds: 180));
      } else {
        await Future<void>.delayed(const Duration(milliseconds: 350));
      }

      return await highResController.takePicture();
    } finally {
      try {
        await highResController.dispose();
      } catch (_) {}
    }
  }

  Future<File> _createMagnifierCrop({
    required String imagePath,
    required DetectionResult detection,
    required DetectionResult fallbackDetection,
    required bool refined,
  }) async {
    final sourceBytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(sourceBytes);
    if (decoded == null) {
      throw StateError('Не удалось прочитать снимок высокого разрешения.');
    }

    final oriented = img.bakeOrientation(decoded);
    final cropRect = _normalizedDetectionToCropRect(
      refined ? detection : fallbackDetection,
      oriented.width,
      oriented.height,
      padding: refined ? 0.20 : 0.30,
    );

    final cropped = img.copyCrop(
      oriented,
      x: cropRect.left.round(),
      y: cropRect.top.round(),
      width: math.max(1, cropRect.width.round()),
      height: math.max(1, cropRect.height.round()),
    );

    final outPath = '${imagePath}_date_magnifier.jpg';
    final outFile = File(outPath);
    await outFile.writeAsBytes(img.encodeJpg(cropped, quality: 95), flush: true);
    return outFile;
  }

  Rect _normalizedDetectionToCropRect(
    DetectionResult detection,
    int imageWidth,
    int imageHeight, {
    required double padding,
  }) {
    final left = detection.left.clamp(0.0, 1.0).toDouble();
    final top = detection.top.clamp(0.0, 1.0).toDouble();
    final right = detection.right.clamp(0.0, 1.0).toDouble();
    final bottom = detection.bottom.clamp(0.0, 1.0).toDouble();

    final centerX = (left + right) / 2.0;
    final centerY = (top + bottom) / 2.0;
    final boxW = math.max(0.02, right - left);
    final boxH = math.max(0.02, bottom - top);

    var cropW = math.min(1.0, boxW * (1.0 + padding * 2.0));
    var cropH = math.min(1.0, boxH * (1.0 + padding * 2.0));

    // Минимальный контекст нужен для маленьких дат: иначе crop получится слишком тонким.
    cropW = math.max(cropW, 0.05);
    cropH = math.max(cropH, 0.025);
    cropW = math.min(cropW, 1.0);
    cropH = math.min(cropH, 1.0);

    final cropLeft = (centerX - cropW / 2.0).clamp(0.0, 1.0 - cropW).toDouble();
    final cropTop = (centerY - cropH / 2.0).clamp(0.0, 1.0 - cropH).toDouble();

    final x = (cropLeft * imageWidth).floor().clamp(0, imageWidth - 1).toInt();
    final y = (cropTop * imageHeight).floor().clamp(0, imageHeight - 1).toInt();
    final w = math.max(1, (cropW * imageWidth).round()).clamp(1, imageWidth - x).toInt();
    final h = math.max(1, (cropH * imageHeight).round()).clamp(1, imageHeight - y).toInt();

    return Rect.fromLTWH(x.toDouble(), y.toDouble(), w.toDouble(), h.toDouble());
  }

  Future<void> _closeMagnifier({bool resumeCamera = true}) async {
    _magnifierEpoch++;
    final imageFile = _magnifierImageFile;

    if (mounted) {
      setState(() {
        _magnifierBusy = false;
        _magnifierImageFile = null;
        _magnifierSourceDetection = null;
        _magnifierRecognizedDate = null;
        _magnifierRecognizedDateText = null;
        _magnifierStatus = '';
      });
    } else {
      _magnifierBusy = false;
      _magnifierImageFile = null;
      _magnifierSourceDetection = null;
      _magnifierRecognizedDate = null;
      _magnifierRecognizedDateText = null;
      _magnifierStatus = '';    }

    try {
      if (imageFile != null && await imageFile.exists()) {
        await imageFile.delete();
      }
    } catch (_) {}

    if (resumeCamera && mounted && !_coveredByChildRoute) {
      await _resumeScanning(recreateCamera: true);
    }
  }

  Future<void> _closeMagnifierForNavigation() async {
    if (!_magnifierActive) return;
    await _closeMagnifier(resumeCamera: false);
  }


  String _formatDateRu(DateTime date) {
    final d = date.day.toString().padLeft(2, '0');
    final m = date.month.toString().padLeft(2, '0');
    return '$d.$m.${date.year}';
  }

  void _showSnackBar(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _openManualAdd({DateTime? initialExpiry, bool keepMagnifier = false}) async {
    if (_manualAddSheetOpen) return;
    if (!widget.auth.isAuthed) {
      if (keepMagnifier) {
        _showSnackBar('Для добавления продукта нужно авторизоваться');
        return;
      }
      _goLogin(after: 'manual');
      return;
    }

    _manualAddSheetOpen = true;
    try {
      if (!keepMagnifier) {
        await _closeMagnifierForNavigation();
        final added = await _showModalWithScanPaused<bool>(
          () => showModalBottomSheet<bool>(
            context: context,
            isScrollControlled: true,
            builder: (_) => ManualAddSheet(
              auth: widget.auth,
              initialExpiry: initialExpiry,
            ),
          ),
        );
        if (added == true && mounted) {
          _showSnackBar('Продукт добавлен в личный кабинет');
        }
        return;
      }

      final added = await showModalBottomSheet<bool>(
        context: context,
        isScrollControlled: true,
        builder: (_) => ManualAddSheet(
          auth: widget.auth,
          initialExpiry: initialExpiry,
        ),
      );
      if (added == true && mounted) {
        _showSnackBar('Продукт добавлен в личный кабинет');
      }
    } finally {
      _manualAddSheetOpen = false;
    }
  }

  Widget _buildMagnifierOverlay() {
    final imageFile = _magnifierImageFile;
    final busy = _magnifierBusy;

    return Positioned.fill(
      child: Container(
        color: Colors.black.withOpacity(0.96),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 84, 16, 88),
            child: Column(
              children: [
                Expanded(
                  child: Center(
                    child: busy || imageFile == null
                        ? Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const CircularProgressIndicator(),
                              const SizedBox(height: 18),
                              Text(
                                _magnifierStatus.isEmpty ? 'Готовлю лупу...' : _magnifierStatus,
                                textAlign: TextAlign.center,
                                style: const TextStyle(color: Colors.white, fontSize: 16),
                              ),
                            ],
                          )
                        : InteractiveViewer(
                            minScale: 1,
                            maxScale: 6,
                            child: Image.file(
                              imageFile,
                              fit: BoxFit.contain,
                              filterQuality: FilterQuality.high,
                            ),
                          ),
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  _magnifierStatus,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white70),
                ),
                if (!busy && _magnifierRecognizedDate != null) ...[
                  const SizedBox(height: 10),
                  FilledButton.icon(
                    onPressed: () => unawaited(
                      _openManualAdd(
                        initialExpiry: _magnifierRecognizedDate,
                        keepMagnifier: true,
                      ),
                    ),
                    icon: const Icon(Icons.add_shopping_cart),
                    label: Text('Добавить продукт с датой ${_formatDateRu(_magnifierRecognizedDate!)}'),
                  ),
                ],
                const SizedBox(height: 12),
                FilledButton.icon(
                  onPressed: busy ? null : () => unawaited(_closeMagnifier()),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Назад к сканированию'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_disposeCameraController(updateState: false));
    _detector.close();
    _dateOcr.close();
    super.dispose();
  }

  void _goLogin({required String after}) async {
    await _closeMagnifierForNavigation();
    await _pushWithScanPaused(
      MaterialPageRoute(builder: (_) => LoginScreen(auth: widget.auth, after: after)),
    );
    if (mounted) setState(() {});
  }

  void _openSettings() async {
    await _closeMagnifierForNavigation();
    await _pushWithScanPaused(
      MaterialPageRoute(builder: (_) => const SettingsScreen()),
    );
  }

  void _onProfile() async {
    if (!widget.auth.isAuthed) return _goLogin(after: 'profile');
    await _closeMagnifierForNavigation();
    await _pushWithScanPaused(
      MaterialPageRoute(builder: (_) => CabinetScreen(auth: widget.auth)),
    );
  }

  void _onManualAdd() async {
    await _openManualAdd(
      initialExpiry: _magnifierActive ? _magnifierRecognizedDate : null,
      keepMagnifier: _magnifierActive,
    );
  }

  void _onError() async {
    if (!widget.auth.isAuthed) return _goLogin(after: 'error');
    await _closeMagnifierForNavigation();
    await _pushWithScanPaused(
      MaterialPageRoute(
        builder: (_) => ErrorDatasetFlowScreen(auth: widget.auth, cameras: widget.cameras),
      ),
      releaseCamera: true,
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
    return PopScope(
      canPop: !_magnifierActive,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) return;
        if (_magnifierActive) {
          await _closeMagnifier();
        }
      },
      child: Scaffold(
        body: Stack(
        children: [
          Positioned.fill(
            child: (() {
              final controller = _controller;
              if (controller == null) {
                return Container(
                    color: Colors.black,
                    alignment: Alignment.center,
                    padding: const EdgeInsets.all(24),
                    child: Text(
                      _aiStatus,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  );
              }

              return FutureBuilder<void>(
                    key: ValueKey('camera-future-$_previewKeySeed'),
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
                        return const ColoredBox(
                          color: Colors.black,
                          child: Center(child: CircularProgressIndicator()),
                        );
                      }

                      final previewSize = controller.value.previewSize;
                      if (previewSize == null) {
                        return const ColoredBox(
                          color: Colors.black,
                          child: Center(child: CircularProgressIndicator()),
                        );
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
                              child: CameraPreview(
                                controller,
                                key: ValueKey('camera-preview-$_previewKeySeed'),
                              ),
                            ),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) {
                              final viewportSize = Size(constraints.maxWidth, constraints.maxHeight);
                              return GestureDetector(
                                behavior: HitTestBehavior.translucent,
                                onTapUp: (details) {
                                  final detection = _hitTestDetection(
                                    details.localPosition,
                                    viewportSize,
                                    childSize,
                                  );
                                  if (detection != null) {
                                    unawaited(_onDetectionTap(detection));
                                  }
                                },
                                child: CustomPaint(
                                  painter: _DetectionOverlayPainter(
                                    detections: _detections,
                                    previewContentSize: childSize,
                                  ),
                                  size: Size.infinite,
                                ),
                              );
                            },
                          ),
                        ],
                      );
                    },
                  );
            })(),
          ),
          if (_magnifierActive) _buildMagnifierOverlay(),
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
          if (!_magnifierActive)
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
