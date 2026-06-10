import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:archive/archive_io.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../api/api_client.dart';
import '../auth/auth_flow.dart';
import '../auth/auth_state.dart';
import '../services/pending_report_repository.dart';
import '../storage/file_key_value_store.dart';

class ErrorDatasetFlowScreen extends StatefulWidget {
  final AuthState auth;
  final List<CameraDescription> cameras;

  const ErrorDatasetFlowScreen({super.key, required this.auth, required this.cameras});

  @override
  State<ErrorDatasetFlowScreen> createState() => _ErrorDatasetFlowScreenState();

  static const int maxSeconds = 30;
  static const int fps = 3; // safer frame capture rate for error reports
}


class _ErrorDatasetFlowScreenState extends State<ErrorDatasetFlowScreen> {
  CameraController? _controller;
  Future<void>? _initFuture;

  Timer? _limitTimer;
  bool _isRecording = false;
  bool _extracting = false;
  bool _uploading = false;
  bool _stopping = false;
  bool _recordingCompleted = false;
  bool _leaving = false;

  Future<void>? _captureLoopFuture;
  int _captureIndex = 0;

  XFile? _videoFile;

  Directory? _workDir;
  Directory? _imagesDir;
  Directory? _labelsDir;

  List<File> _frames = [];
  int _index = 0;

  // bboxes per frame in real image pixel coords. One frame can contain several date areas.
  final Map<int, List<Rect>> _bboxes = {};
  final Set<int> _skipped = {};
  final Map<int, Size> _frameSizes = {};
  String? _validationToken;
  String? _validationFrameName;
  int? _validationFrameIndex;

  final _comment = TextEditingController();
  final _pendingReports = PendingReportRepository();
  String? _status;

  @override
  void initState() {
    super.initState();

    final back = widget.cameras.where((c) => c.lensDirection == CameraLensDirection.back);
    final cam = back.isNotEmpty ? back.first : widget.cameras.first;

    // Audio is disabled intentionally: error reports need frames only.
    // This avoids RECORD_AUDIO permission issues and native crashes on some devices.
    _controller = CameraController(cam, ResolutionPreset.medium, enableAudio: false);
    _initFuture = _controller!.initialize();
  }

  Future<void> _releaseCamera() async {
    _isRecording = false;
    final controller = _controller;
    _controller = null;
    _initFuture = null;

    if (controller == null) return;

    try {
      if (controller.value.isInitialized && controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (_) {
      // Ignore camera stream races during route disposal.
    }

    try {
      await controller.dispose();
    } catch (_) {
      // Ignore double-dispose/native disposal races.
    }
  }

  Future<bool> _prepareToLeave() async {
    if (_uploading || _stopping) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Дождись завершения текущей операции')),
        );
      }
      return false;
    }

    if (_leaving) return false;
    _leaving = true;
    _limitTimer?.cancel();
    _isRecording = false;

    if (mounted) {
      setState(() => _status = 'Освобождаю камеру...');
    }

    final loop = _captureLoopFuture;
    if (loop != null) {
      try {
        await loop.timeout(const Duration(seconds: 5), onTimeout: () {});
      } catch (_) {
        // The loop can fail if native camera is already being disposed.
      }
    }

    await _releaseCamera();
    return true;
  }

  @override
  void dispose() {
    _limitTimer?.cancel();
    _isRecording = false;
    unawaited(_releaseCamera());
    _comment.dispose();
    super.dispose();
  }

  Future<void> _startRecording() async {
    try {
      await _initFuture;
      if (_controller == null || !_controller!.value.isInitialized) {
        setState(() => _status = 'Камера ещё не готова.');
        return;
      }

      _limitTimer?.cancel();
      await _prepareCaptureDirectories();

      setState(() {
        _isRecording = true;
        _stopping = false;
        _extracting = false;
        _recordingCompleted = false;
        _videoFile = null;
        _frames = [];
        _index = 0;
        _bboxes.clear();
        _skipped.clear();
        _frameSizes.clear();
        _validationToken = null;
        _validationFrameName = null;
        _validationFrameIndex = null;
        _status = 'Запись кадров... (макс. ${ErrorDatasetFlowScreen.maxSeconds}s)';
      });

      _captureIndex = 0;
      _captureLoopFuture = _captureFramesLoop();

      _limitTimer = Timer(const Duration(seconds: ErrorDatasetFlowScreen.maxSeconds), () {
        if (mounted && _isRecording) {
          _stopRecording();
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isRecording = false;
        _stopping = false;
        _extracting = false;
        _status = 'Не удалось начать запись: $e';
      });
    }
  }

  Future<void> _prepareCaptureDirectories() async {
    final reportsDir = await FileKeyValueStore.namedDirectory('error_reports');
    final root = Directory('${reportsDir.path}/error_report_${DateTime.now().millisecondsSinceEpoch}');
    await root.create(recursive: true);

    final images = Directory('${root.path}/images');
    final labels = Directory('${root.path}/labels');
    await images.create(recursive: true);
    await labels.create(recursive: true);

    _workDir = root;
    _imagesDir = images;
    _labelsDir = labels;
  }

  Future<void> _captureFramesLoop() async {
    final interval = Duration(milliseconds: (1000 / ErrorDatasetFlowScreen.fps).round());

    while (mounted && _isRecording) {
      try {
        final controller = _controller;
        final images = _imagesDir;
        if (controller == null || images == null || !controller.value.isInitialized) {
          break;
        }

        final shot = await controller.takePicture();
        final index = _captureIndex++;
        final frameFile = File('${images.path}/${_frameName(index)}');

        await _normalizeCameraJpeg(File(shot.path), frameFile);

        if (!mounted) return;
        setState(() {
          _frames.add(frameFile);
          _status = 'Запись кадров: ${_frames.length}... Нажми «Стоп», когда ошибка попала в кадр.';
        });
      } catch (e) {
        if (!mounted) return;
        setState(() => _status = 'Не удалось сохранить один кадр, продолжаю: $e');
        await Future.delayed(const Duration(milliseconds: 700));
      }

      await Future.delayed(interval);
    }
  }

  Future<void> _normalizeCameraJpeg(File source, File destination) async {
    try {
      final bytes = await source.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) {
        await source.copy(destination.path);
      } else {
        // Android camera JPEGs can contain EXIF orientation. Flutter may display the
        // rotated image while backend/training reads raw pixels, which makes boxes
        // appear shifted. Baking orientation makes preview pixels and saved labels
        // use the same coordinate system.
        final normalized = img.bakeOrientation(decoded);
        await destination.writeAsBytes(img.encodeJpg(normalized, quality: 92), flush: true);
      }
    } finally {
      try {
        if (source.path != destination.path && await source.exists()) {
          await source.delete();
        }
      } catch (_) {
        // Temporary camera files are cleaned by the OS if deletion is unavailable.
      }
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording || _stopping) return;

    _limitTimer?.cancel();
    setState(() {
      _isRecording = false;
      _stopping = true;
      _extracting = true;
      _status = 'Останавливаю запись и подготавливаю кадры...';
    });

    try {
      final loop = _captureLoopFuture;
      if (loop != null) {
        await loop.timeout(const Duration(seconds: 12), onTimeout: () {});
      }

      if (_frames.isEmpty) {
        throw Exception('Не удалось сохранить ни одного кадра. Попробуй записать чуть дольше.');
      }

      await _tryInjectValidationFrame();

      final sizes = <int, Size>{};
      for (var i = 0; i < _frames.length; i++) {
        sizes[i] = await _readImageSize(_frames[i]);
      }

      await _releaseCamera();

      if (!mounted) return;
      setState(() {
        _frameSizes
          ..clear()
          ..addAll(sizes);
        _index = 0;
        _recordingCompleted = true;
        _status = 'Кадры готовы: ${_frames.length}. Размечай по очереди.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _recordingCompleted = false;
        _status = 'Ошибка после остановки записи: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _stopping = false;
          _extracting = false;
        });
      }
    }
  }

  bool get _allAnnotated =>
      _frames.isNotEmpty &&
      List<int>.generate(_frames.length, (i) => i).every((i) {
        if (_skipped.contains(i)) return true;
        final boxes = _bboxes[i];
        return boxes != null && boxes.isNotEmpty;
      });

  int get _usableFramesCount => _frames.length - _skipped.length;

  String _frameName(int index) {
    final i = index + 1;
    return 'frame_${i.toString().padLeft(5, '0')}.jpg';
  }

  String _imageNameForIndex(int index) => _frames[index].uri.pathSegments.last;

  String _labelNameForIndex(int index) {
    final imageName = _imageNameForIndex(index);
    final dot = imageName.lastIndexOf('.');
    if (dot <= 0) return '$imageName.txt';
    return '${imageName.substring(0, dot)}.txt';
  }

  Future<void> _tryInjectValidationFrame() async {
    final images = _imagesDir;
    if (images == null || _frames.isEmpty) return;

    try {
      final challenge = await AuthFlow.runWithReauth<Map<String, dynamic>?>(
        context: context,
        auth: widget.auth,
        after: 'error',
        action: () => ApiClient(token: widget.auth.token).reportValidationFrame(),
      );

      if (challenge == null) return;

      final token = challenge['validationToken']?.toString() ?? '';
      final fileNameRaw = challenge['fileName']?.toString() ?? '';
      final imageBase64 = challenge['imageBase64']?.toString() ?? '';
      if (token.isEmpty || fileNameRaw.isEmpty || imageBase64.isEmpty) return;

      final safeFileName = fileNameRaw.split('/').last.split('\\').last;
      final lower = safeFileName.toLowerCase();
      if (!lower.endsWith('.jpg') && !lower.endsWith('.jpeg') && !lower.endsWith('.png')) return;

      final file = File('${images.path}/$safeFileName');
      await file.writeAsBytes(base64Decode(imageBase64), flush: true);

      final insertAt = math.Random().nextInt(_frames.length + 1);
      _frames.insert(insertAt, file);
      _validationToken = token;
      _validationFrameName = safeFileName;
      _validationFrameIndex = insertAt;
    } on NetworkApiException {
      // Отчёт можно подготовить офлайн. Проверочный кадр будет добавлен только при наличии связи.
    } on ApiRequestException {
      // Если сервер временно не нашёл подходящий подтверждённый кадр, не блокируем создание отчёта.
    }
  }

  Future<Size> _readImageSize(File file) async {
    final bytes = await file.readAsBytes();
    final completer = Completer<Size>();
    ui.decodeImageFromList(bytes, (image) {
      completer.complete(Size(image.width.toDouble(), image.height.toDouble()));
    });
    return completer.future;
  }

  Future<Size> _frameSizeForIndex(int index) async {
    final cached = _frameSizes[index];
    if (cached != null) return cached;
    final size = await _readImageSize(_frames[index]);
    _frameSizes[index] = size;
    return size;
  }

  Future<void> _writeYoloLabel(int index, List<Rect> rects) async {
    final labels = _labelsDir!;
    final name = _labelNameForIndex(index);
    final path = '${labels.path}/$name';

    final imageSize = await _frameSizeForIndex(index);
    final width = imageSize.width;
    final height = imageSize.height;

    double clampX(double v) => v < 0 ? 0 : (v > width ? width : v);
    double clampY(double v) => v < 0 ? 0 : (v > height ? height : v);

    final lines = <String>[];
    for (final r in rects) {
      final left = clampX(r.left);
      final top = clampY(r.top);
      final right = clampX(r.right);
      final bottom = clampY(r.bottom);

      final w = right - left;
      final h = bottom - top;
      if (w < 2 || h < 2) continue;

      final cx = left + w / 2.0;
      final cy = top + h / 2.0;

      final xc = cx / width;
      final yc = cy / height;
      final wn = w / width;
      final hn = h / height;

      lines.add('0 ${xc.toStringAsFixed(6)} ${yc.toStringAsFixed(6)} ${wn.toStringAsFixed(6)} ${hn.toStringAsFixed(6)}');
    }

    if (lines.isEmpty) {
      throw Exception('На кадре нет валидных областей разметки.');
    }

    final content = lines.join('\n') + '\n';
    await File(path).writeAsString(content, flush: true);
  }

  Future<void> _saveCurrentAndNext() async {
    final boxes = _bboxes[_index];
    if (boxes == null || boxes.isEmpty) return;

    _skipped.remove(_index);
    await _writeYoloLabel(_index, boxes);

    if (_index < _frames.length - 1) {
      setState(() => _index++);
    } else {
      setState(() => _status = 'Разметка завершена. Можно отправлять.');
    }
  }

  Future<void> _skipCurrentFrame() async {
    final labels = _labelsDir;
    final current = _index;

    _bboxes.remove(current);
    _skipped.add(current);

    if (labels != null) {
      final labelFile = File('${labels.path}/${_labelNameForIndex(current)}');
      try {
        if (await labelFile.exists()) await labelFile.delete();
      } catch (_) {
        // If deletion fails, the ZIP step still ignores skipped frames.
      }
    }

    if (!mounted) return;
    if (current < _frames.length - 1) {
      setState(() {
        _index++;
        _status = 'Кадр ${current + 1} пропущен.';
      });
    } else {
      setState(() => _status = 'Последний кадр пропущен. Можно отправлять, если есть размеченные кадры.');
    }
  }

  Future<String> _zipDataset() async {
    final root = _workDir!;
    final zipPath = '${root.path}/dataset.zip';

    if (_usableFramesCount <= 0) {
      throw Exception('Все кадры пропущены. Оставь хотя бы один размеченный кадр.');
    }

    for (var i = 0; i < _frames.length; i++) {
      if (_skipped.contains(i)) continue;

      final boxes = _bboxes[i];
      if (boxes == null || boxes.isEmpty) {
        throw Exception('Не все кадры размечены или пропущены.');
      }

      await _writeYoloLabel(i, boxes);
    }

    final imageFiles = <File>[];
    final labelFiles = <File>[];
    for (var i = 0; i < _frames.length; i++) {
      if (_skipped.contains(i)) continue;
      imageFiles.add(_frames[i]);
      labelFiles.add(File('${_labelsDir!.path}/${_labelNameForIndex(i)}')); 
    }

    debugPrint('ZIP INPUT IMAGES FINAL: ${imageFiles.length}');
    debugPrint('ZIP INPUT LABELS FINAL: ${labelFiles.length}');

    final encoder = ZipFileEncoder();
    encoder.create(zipPath);

    for (final f in imageFiles) {
      encoder.addFile(f, 'images/${f.uri.pathSegments.last}');
    }

    for (final f in labelFiles) {
      encoder.addFile(f, 'labels/${f.uri.pathSegments.last}');
    }

    final meta = File('${root.path}/meta.json');
    await meta.writeAsString(
      jsonEncode({
        'fps': ErrorDatasetFlowScreen.fps,
        'frames': _frames.length,
        'preserveAspect': true,
        'validationFrameName': _validationFrameName,
      }),
      flush: true,
    );
    encoder.addFile(meta, 'meta.json');

    encoder.close();

    final zipCheck = ZipDecoder().decodeBytes(await File(zipPath).readAsBytes());
    debugPrint('ZIP ENTRIES COUNT: ${zipCheck.files.length}');
    for (final entry in zipCheck.files) {
      debugPrint('ZIP ENTRY: ${entry.name}');
    }
    return zipPath;
  }

  Future<void> _upload() async {
    if (!_allAnnotated) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Разметь или пропусти все кадры')));
      return;
    }

    if (_usableFramesCount <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Нельзя отправить отчёт без размеченных кадров')));
      return;
    }

    setState(() {
      _uploading = true;
      _status = 'Архивация и отправка...';
    });

    try {
      final zipPath = await _zipDataset();
      final comment = _comment.text.trim();

      try {
        await AuthFlow.runWithReauth<void>(
          context: context,
          auth: widget.auth,
          after: 'error',
          action: () => ApiClient(token: widget.auth.token).uploadDatasetZip(
            zipPath,
            comment: comment,
            validationToken: _validationToken,
            validationFrameName: _validationFrameName,
          ),
        );
      } on NetworkApiException {
        await _pendingReports.addPendingReport(
          zipPath: zipPath,
          comment: comment,
          validationToken: _validationToken,
          validationFrameName: _validationFrameName,
        );
        await _releaseCamera();
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Интернета нет. Отчёт сохранён и отправится позже.')),
        );
        Navigator.of(context).pop();
        return;
      }

      await _releaseCamera();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Отправлено ✅')));
      Navigator.of(context).pop();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Ошибка: $e')));
      setState(() => _status = 'Ошибка: $e');
    } finally {
      if (mounted) setState(() => _uploading = false);
    }
  }

  void _setBBoxesForCurrent(List<Rect> rects) {
    setState(() {
      if (rects.isEmpty) {
        _bboxes.remove(_index);
      } else {
        _skipped.remove(_index);
        _bboxes[_index] = List<Rect>.unmodifiable(rects);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final readyToAnnotate = _recordingCompleted && _frames.isNotEmpty;
    final currentFile = readyToAnnotate ? _frames[_index] : null;

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) return;
        final ok = await _prepareToLeave();
        if (ok && context.mounted) {
          Navigator.of(context).pop(result);
        }
      },
      child: Scaffold(
        appBar: AppBar(title: const Text('Сообщить об ошибке')),
        body: FutureBuilder(
          future: _initFuture,
          builder: (context, snap) {
            final controller = _controller;
            if (!readyToAnnotate && (controller == null || _initFuture == null)) {
              return Center(child: Text(_status ?? 'Камера освобождается...'));
            }

            if (!readyToAnnotate && snap.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }

            return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              if (!readyToAnnotate) ...[
                _ReportCameraPreview(controller: controller!),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: (_extracting || _isRecording || _stopping) ? null : _startRecording,
                        icon: const Icon(Icons.fiber_manual_record),
                        label: const Text('Начать запись'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: (_isRecording && !_stopping) ? _stopRecording : null,
                        icon: const Icon(Icons.stop),
                        label: const Text('Стоп'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                const Text('Лимит записи: 30 секунд. Приложение сохраняет кадры напрямую, без FFmpeg, чтобы не падать на stop.'),
                const SizedBox(height: 8),
                if (_status != null) Text(_status!),
                if (_extracting) const Padding(padding: EdgeInsets.only(top: 12), child: LinearProgressIndicator()),
              ] else ...[
                Text('Кадр ${_index + 1} / ${_frames.length}' + (_skipped.contains(_index) ? ' — пропущен' : '')),
                const SizedBox(height: 12),
                _Annotator(
                  imageFile: currentFile!,
                  initial: _bboxes[_index] ?? const [],
                  onChanged: _setBBoxesForCurrent,
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _comment,
                  decoration: const InputDecoration(
                    labelText: 'Комментарий (опционально)',
                    border: OutlineInputBorder(),
                  ),
                  minLines: 2,
                  maxLines: 4,
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: _index > 0 ? () => setState(() => _index--) : null,
                        child: const Text('Назад'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: _skipCurrentFrame,
                        icon: const Icon(Icons.skip_next),
                        label: const Text('Пропустить'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: (_bboxes[_index]?.isNotEmpty ?? false) ? _saveCurrentAndNext : null,
                    child: Text(_index == _frames.length - 1 ? 'Завершить' : 'Сохранить и далее'),
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton.icon(
                  onPressed: (_uploading || !_allAnnotated || _usableFramesCount <= 0) ? null : _upload,
                  icon: const Icon(Icons.upload),
                  label: Text(_uploading ? 'Отправка...' : 'Отправить'),
                ),
                const SizedBox(height: 8),
                Text('Размечено кадров: ${_bboxes.length}, областей: ${_bboxes.values.fold<int>(0, (sum, boxes) => sum + boxes.length)}, пропущено: ${_skipped.length}, всего: ${_frames.length}'),
                if (_status != null) Text(_status!),
              ],
            ],
          );
        },
        ),
      ),
    );
  }
}


class _ReportCameraPreview extends StatelessWidget {
  final CameraController controller;

  const _ReportCameraPreview({required this.controller});

  @override
  Widget build(BuildContext context) {
    final previewSize = controller.value.previewSize;
    if (previewSize == null) {
      return const AspectRatio(
        aspectRatio: 3 / 4,
        child: ColoredBox(
          color: Colors.black,
          child: Center(child: CircularProgressIndicator()),
        ),
      );
    }

    final childSize = Size(previewSize.height, previewSize.width);
    return AspectRatio(
      aspectRatio: childSize.width / childSize.height,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: childSize.width,
            height: childSize.height,
            child: CameraPreview(
              controller,
              key: ValueKey('report-camera-${controller.hashCode}'),
            ),
          ),
        ),
      ),
    );
  }
}

class _Annotator extends StatefulWidget {
  final File imageFile;
  final List<Rect> initial;
  final ValueChanged<List<Rect>> onChanged;

  const _Annotator({required this.imageFile, required this.initial, required this.onChanged});

  @override
  State<_Annotator> createState() => _AnnotatorState();
}

class _AnnotatorState extends State<_Annotator> {
  Offset? _start;
  Rect? _draftRect;
  late List<Rect> _rects;
  late Future<Size> _imageSizeFuture;

  @override
  void initState() {
    super.initState();
    _rects = List<Rect>.from(widget.initial);
    _imageSizeFuture = _readImageSize(widget.imageFile);
  }

  @override
  void didUpdateWidget(covariant _Annotator oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.imageFile.path != widget.imageFile.path) {
      _rects = List<Rect>.from(widget.initial);
      _draftRect = null;
      _start = null;
      _imageSizeFuture = _readImageSize(widget.imageFile);
    } else if (!_sameRects(oldWidget.initial, widget.initial)) {
      _rects = List<Rect>.from(widget.initial);
      _draftRect = null;
    }
  }

  bool _sameRects(List<Rect> a, List<Rect> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  Future<Size> _readImageSize(File file) async {
    final bytes = await file.readAsBytes();
    final completer = Completer<Size>();
    ui.decodeImageFromList(bytes, (image) {
      completer.complete(Size(image.width.toDouble(), image.height.toDouble()));
    });
    return completer.future;
  }

  Rect _rectFrom(Offset a, Offset b) {
    final left = math.min(a.dx, b.dx);
    final top = math.min(a.dy, b.dy);
    final right = math.max(a.dx, b.dx);
    final bottom = math.max(a.dy, b.dy);
    return Rect.fromLTRB(left, top, right, bottom);
  }

  void _publish() => widget.onChanged(List<Rect>.unmodifiable(_rects));

  void _clearAll() {
    setState(() {
      _rects.clear();
      _draftRect = null;
      _start = null;
    });
    _publish();
  }

  void _undoLast() {
    if (_rects.isEmpty) return;
    setState(() => _rects.removeLast());
    _publish();
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Size>(
      future: _imageSizeFuture,
      builder: (context, snap) {
        if (!snap.hasData) {
          return const AspectRatio(
            aspectRatio: 1,
            child: Center(child: CircularProgressIndicator()),
          );
        }

        final imageSize = snap.data!;
        return LayoutBuilder(
          builder: (context, constraints) {
            final maxWidth = math.min(constraints.maxWidth, 360.0);
            final aspectRatio = imageSize.width / imageSize.height;
            final displayWidth = maxWidth;
            final displayHeight = displayWidth / aspectRatio;
            final scaleX = imageSize.width / displayWidth;
            final scaleY = imageSize.height / displayHeight;

            Offset clampLocal(Offset local) => Offset(
                  local.dx.clamp(0.0, displayWidth).toDouble(),
                  local.dy.clamp(0.0, displayHeight).toDouble(),
                );

            Offset toImage(Offset local) {
              final clamped = clampLocal(local);
              return Offset(clamped.dx * scaleX, clamped.dy * scaleY);
            }

            Rect toLocalRect(Rect r) => Rect.fromLTRB(
                  r.left / scaleX,
                  r.top / scaleY,
                  r.right / scaleX,
                  r.bottom / scaleY,
                );

            final localRects = _rects.map(toLocalRect).toList(growable: false);
            final localDraft = _draftRect == null ? null : toLocalRect(_draftRect!);

            return Center(
              child: SizedBox(
                width: displayWidth,
                height: displayHeight,
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: GestureDetector(
                        onPanStart: (d) {
                          _start = toImage(d.localPosition);
                          setState(() => _draftRect = Rect.fromLTWH(_start!.dx, _start!.dy, 1, 1));
                        },
                        onPanUpdate: (d) {
                          if (_start == null) return;
                          final cur = toImage(d.localPosition);
                          setState(() => _draftRect = _rectFrom(_start!, cur));
                        },
                        onPanEnd: (_) {
                          final draft = _draftRect;
                          if (draft != null && draft.width >= 4 && draft.height >= 4) {
                            setState(() {
                              _rects.add(draft);
                              _draftRect = null;
                            });
                            _publish();
                          } else {
                            setState(() => _draftRect = null);
                          }
                          _start = null;
                        },
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.file(widget.imageFile, fit: BoxFit.fill),
                        ),
                      ),
                    ),
                    Positioned.fill(
                      child: IgnorePointer(
                        child: CustomPaint(painter: _RectPainter(rects: localRects, draft: localDraft)),
                      ),
                    ),
                    Positioned(
                      left: 8,
                      top: 8,
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.55),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                          child: Text(
                            'Областей: ${_rects.length}',
                            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      right: 8,
                      top: 8,
                      child: Wrap(
                        spacing: 6,
                        children: [
                          FilledButton.tonal(
                            onPressed: _rects.isEmpty ? null : _undoLast,
                            child: const Text('Назад'),
                          ),
                          FilledButton.tonal(
                            onPressed: _rects.isEmpty && _draftRect == null ? null : _clearAll,
                            child: const Text('Сброс'),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }
}

class _RectPainter extends CustomPainter {
  final List<Rect> rects;
  final Rect? draft;

  _RectPainter({required this.rects, required this.draft});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = const Color(0xFF1565C0);

    for (final rect in rects) {
      canvas.drawRect(rect, paint);
    }

    final draftRect = draft;
    if (draftRect != null) {
      final draftPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = const Color(0xFF00C853);
      canvas.drawRect(draftRect, draftPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _RectPainter oldDelegate) =>
      oldDelegate.draft != draft || !_sameRects(oldDelegate.rects, rects);

  bool _sameRects(List<Rect> a, List<Rect> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
