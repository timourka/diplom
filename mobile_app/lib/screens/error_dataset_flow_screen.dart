import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:archive/archive_io.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';

import '../api/api_client.dart';
import '../auth/auth_state.dart';

class ErrorDatasetFlowScreen extends StatefulWidget {
  final AuthState auth;
  final List<CameraDescription> cameras;

  const ErrorDatasetFlowScreen({super.key, required this.auth, required this.cameras});

  @override
  State<ErrorDatasetFlowScreen> createState() => _ErrorDatasetFlowScreenState();

  static const int maxSeconds = 30;
  static const int fps = 6; // ~each 5th frame from 30fps
  static const int imgSize = 640; // 640x640 for YOLO
}

class _ErrorDatasetFlowScreenState extends State<ErrorDatasetFlowScreen> {
  CameraController? _controller;
  Future<void>? _initFuture;

  Timer? _limitTimer;
  bool _isRecording = false;
  bool _extracting = false;
  bool _uploading = false;

  XFile? _videoFile;

  Directory? _workDir;
  Directory? _imagesDir;
  Directory? _labelsDir;

  List<File> _frames = [];
  int _index = 0;

  // bbox per frame in image pixel coords (0..640)
  final Map<int, Rect> _bboxes = {};

  final _comment = TextEditingController();
  String? _status;

  @override
  void initState() {
    super.initState();

    final back = widget.cameras.where((c) => c.lensDirection == CameraLensDirection.back);
    final cam = back.isNotEmpty ? back.first : widget.cameras.first;

    _controller = CameraController(cam, ResolutionPreset.high, enableAudio: true);
    _initFuture = _controller!.initialize();
  }

  @override
  void dispose() {
    _limitTimer?.cancel();
    _controller?.dispose();
    _comment.dispose();
    super.dispose();
  }

  Future<void> _startRecording() async {
    await _initFuture;
    if (_controller == null) return;

    await _controller!.startVideoRecording();

    _limitTimer?.cancel();
    _limitTimer = Timer(const Duration(seconds: ErrorDatasetFlowScreen.maxSeconds), () async {
      if (_isRecording) await _stopRecording();
    });

    setState(() {
      _isRecording = true;
      _status = 'Запись... (макс. ${ErrorDatasetFlowScreen.maxSeconds}s)';
    });
  }

  Future<void> _stopRecording() async {
    if (_controller == null) return;
    _limitTimer?.cancel();

    final file = await _controller!.stopVideoRecording();
    setState(() {
      _isRecording = false;
      _videoFile = file;
      _status = 'Видео записано. Извлекаю кадры...';
    });

    await _prepareDataset();
  }

  Future<void> _prepareDataset() async {
    if (_videoFile == null) return;

    setState(() {
      _extracting = true;
      _status =
          'Извлечение кадров (fps=${ErrorDatasetFlowScreen.fps}, ${ErrorDatasetFlowScreen.imgSize}x${ErrorDatasetFlowScreen.imgSize})...';
    });

    try {
      final appDir = await getApplicationDocumentsDirectory();
      final root = Directory('${appDir.path}/error_report_${DateTime.now().millisecondsSinceEpoch}');
      await root.create(recursive: true);

      final images = Directory('${root.path}/images');
      final labels = Directory('${root.path}/labels');
      await images.create(recursive: true);
      await labels.create(recursive: true);

      _workDir = root;
      _imagesDir = images;
      _labelsDir = labels;

      final input = _videoFile!.path;
      final outPattern = '${images.path}/frame_%05d.jpg';

      final imgSize = ErrorDatasetFlowScreen.imgSize;
      final fps = ErrorDatasetFlowScreen.fps;

      // fps + scale to 640 with letterbox pad to 640x640
      final cmd =
          '-i "$input" -vf "fps=$fps,scale=$imgSize:$imgSize:force_original_aspect_ratio=decrease,pad=$imgSize:$imgSize:(ow-iw)/2:(oh-ih)/2" -q:v 3 "$outPattern"';

      final session = await FFmpegKit.execute(cmd);
      final rc = await session.getReturnCode();
      if (!ReturnCode.isSuccess(rc)) {
        final logs = await session.getAllLogsAsString();
        throw Exception('FFmpeg error: $logs');
      }

      final files = images
          .listSync()
          .whereType<File>()
          .where((f) => f.path.toLowerCase().endsWith('.jpg'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

      if (files.isEmpty) {
        throw Exception('Не удалось извлечь кадры.');
      }

      setState(() {
        _frames = files;
        _index = 0;
        _status = 'Кадры готовы: ${files.length}. Размечай по очереди.';
      });
    } finally {
      setState(() => _extracting = false);
    }
  }

  bool get _allAnnotated => _frames.isNotEmpty && _bboxes.length == _frames.length;

  String _frameName(int index) {
    final i = index + 1;
    return 'frame_${i.toString().padLeft(5, '0')}.jpg';
  }

  Future<void> _writeYoloLabel(int index, Rect r) async {
    final labels = _labelsDir!;
    final name = _frameName(index).replaceAll('.jpg', '.txt');
    final path = '${labels.path}/$name';

    final s = ErrorDatasetFlowScreen.imgSize.toDouble();

    double clamp(double v) => v < 0 ? 0 : (v > s ? s : v);

    final left = clamp(r.left);
    final top = clamp(r.top);
    final right = clamp(r.right);
    final bottom = clamp(r.bottom);

    final w = math.max(1.0, right - left);
    final h = math.max(1.0, bottom - top);
    final cx = left + w / 2.0;
    final cy = top + h / 2.0;

    final xc = cx / s;
    final yc = cy / s;
    final wn = w / s;
    final hn = h / s;

    final line =
        '0 ${xc.toStringAsFixed(6)} ${yc.toStringAsFixed(6)} ${wn.toStringAsFixed(6)} ${hn.toStringAsFixed(6)}\n';
    await File(path).writeAsString(line, flush: true);
  }

  Future<void> _saveCurrentAndNext() async {
    final r = _bboxes[_index];
    if (r == null) return;

    await _writeYoloLabel(_index, r);

    if (_index < _frames.length - 1) {
      setState(() => _index++);
    } else {
      setState(() => _status = 'Разметка завершена. Можно отправлять.');
    }
  }

  Future<String> _zipDataset() async {
    final root = _workDir!;
    final zipPath = '${root.path}/dataset.zip';

    // Ensure all labels exist
    for (var i = 0; i < _frames.length; i++) {
      if (!_bboxes.containsKey(i)) throw Exception('Не все кадры размечены.');
      final txt = File('${_labelsDir!.path}/${_frameName(i).replaceAll('.jpg', '.txt')}');
      if (!await txt.exists()) {
        await _writeYoloLabel(i, _bboxes[i]!);
      }
    }

    final encoder = ZipFileEncoder();
    encoder.create(zipPath);

    encoder.addDirectory(_imagesDir!, includeDirName: true); // images/
    encoder.addDirectory(_labelsDir!, includeDirName: true); // labels/

    final meta = File('${root.path}/meta.json');
    await meta.writeAsString(
      '{"fps":${ErrorDatasetFlowScreen.fps},"imgSize":${ErrorDatasetFlowScreen.imgSize},"frames":${_frames.length}}',
      flush: true,
    );
    encoder.addFile(meta, 'meta.json');
    encoder.close();

    return zipPath;
  }

  Future<void> _upload() async {
    if (!_allAnnotated) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Разметь все кадры')));
      return;
    }

    setState(() {
      _uploading = true;
      _status = 'Архивация и отправка...';
    });

    try {
      final zipPath = await _zipDataset();
      final api = ApiClient(token: widget.auth.token);
      await api.uploadDatasetZip(zipPath, comment: _comment.text.trim());

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Отправлено ✅')));
      Navigator.pop(context);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Ошибка: $e')));
      setState(() => _status = 'Ошибка: $e');
    } finally {
      if (mounted) setState(() => _uploading = false);
    }
  }

  void _setBBoxForCurrent(Rect rect) {
    setState(() => _bboxes[_index] = rect);
  }

  @override
  Widget build(BuildContext context) {
    final readyToAnnotate = _frames.isNotEmpty;
    final currentFile = readyToAnnotate ? _frames[_index] : null;

    return Scaffold(
      appBar: AppBar(title: const Text('Сообщить об ошибке')),
      body: FutureBuilder(
        future: _initFuture,
        builder: (context, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              if (!readyToAnnotate) ...[
                AspectRatio(
                  aspectRatio: _controller!.value.aspectRatio,
                  child: CameraPreview(_controller!),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: (_extracting || _isRecording) ? null : _startRecording,
                        icon: const Icon(Icons.fiber_manual_record),
                        label: const Text('Начать запись'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isRecording ? _stopRecording : null,
                        icon: const Icon(Icons.stop),
                        label: const Text('Стоп'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                const Text('Лимит записи: 30 секунд. Затем нарезаем кадры (fps=6) и приводим к 640x640.'),
                const SizedBox(height: 8),
                if (_status != null) Text(_status!),
                if (_extracting) const Padding(padding: EdgeInsets.only(top: 12), child: LinearProgressIndicator()),
              ] else ...[
                Text('Кадр ${_index + 1} / ${_frames.length}'),
                const SizedBox(height: 12),
                _Annotator(
                  imageFile: currentFile!,
                  initial: _bboxes[_index],
                  onChanged: _setBBoxForCurrent,
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
                    const SizedBox(width: 12),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: _bboxes[_index] == null ? null : _saveCurrentAndNext,
                        child: Text(_index == _frames.length - 1 ? 'Завершить' : 'Сохранить и далее'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                ElevatedButton.icon(
                  onPressed: (_uploading || !_allAnnotated) ? null : _upload,
                  icon: const Icon(Icons.upload),
                  label: Text(_uploading ? 'Отправка...' : 'Отправить'),
                ),
                const SizedBox(height: 8),
                Text('Размечено: ${_bboxes.length}/${_frames.length}'),
                if (_status != null) Text(_status!),
              ],
            ],
          );
        },
      ),
    );
  }
}

class _Annotator extends StatefulWidget {
  final File imageFile;
  final Rect? initial;
  final ValueChanged<Rect> onChanged;

  const _Annotator({required this.imageFile, required this.initial, required this.onChanged});

  @override
  State<_Annotator> createState() => _AnnotatorState();
}

class _AnnotatorState extends State<_Annotator> {
  Offset? _start;
  Rect? _rect;

  @override
  void initState() {
    super.initState();
    _rect = widget.initial;
  }

  @override
  void didUpdateWidget(covariant _Annotator oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.initial != widget.initial) {
      _rect = widget.initial;
    }
  }

  Rect _rectFrom(Offset a, Offset b) {
    final left = math.min(a.dx, b.dx);
    final top = math.min(a.dy, b.dy);
    final right = math.max(a.dx, b.dx);
    final bottom = math.max(a.dy, b.dy);
    return Rect.fromLTRB(left, top, right, bottom);
  }

  @override
  Widget build(BuildContext context) {
    // images are 640x640; we show them in a square and scale gestures back to image pixels.
    return LayoutBuilder(
      builder: (context, constraints) {
        final box = math.min(constraints.maxWidth, 360.0);
        final scale = ErrorDatasetFlowScreen.imgSize / box; // local -> image px

        Offset toImage(Offset local) => Offset(local.dx * scale, local.dy * scale);

        Rect? rectLocal = _rect == null
            ? null
            : Rect.fromLTRB(_rect!.left / scale, _rect!.top / scale, _rect!.right / scale, _rect!.bottom / scale);

        return Center(
          child: GestureDetector(
            onPanStart: (d) {
              _start = toImage(d.localPosition);
              setState(() => _rect = Rect.fromLTWH(_start!.dx, _start!.dy, 1, 1));
            },
            onPanUpdate: (d) {
              if (_start == null) return;
              final cur = toImage(d.localPosition);
              setState(() => _rect = _rectFrom(_start!, cur));
            },
            onPanEnd: (_) {
              if (_rect != null) widget.onChanged(_rect!);
              _start = null;
            },
            child: SizedBox(
              width: box,
              height: box,
              child: Stack(
                children: [
                  Positioned.fill(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(widget.imageFile, fit: BoxFit.fill),
                    ),
                  ),
                  Positioned.fill(child: CustomPaint(painter: _RectPainter(rectLocal))),
                  Positioned(
                    right: 8,
                    top: 8,
                    child: FilledButton.tonal(
                      onPressed: () => setState(() => _rect = null),
                      child: const Text('Сброс'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class _RectPainter extends CustomPainter {
  final Rect? rect;
  _RectPainter(this.rect);

  @override
  void paint(Canvas canvas, Size size) {
    if (rect == null) return;
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;
    canvas.drawRect(rect!, paint);
  }

  @override
  bool shouldRepaint(covariant _RectPainter oldDelegate) => oldDelegate.rect != rect;
}
