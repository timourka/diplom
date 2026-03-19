class DetectionResult {
  final double confidence;
  final int classIndex;
  final double left;
  final double top;
  final double right;
  final double bottom;

  const DetectionResult({
    required this.confidence,
    required this.classIndex,
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  String get label => classIndex == 0 ? 'Дата' : 'Класс $classIndex';
}

class FramePerf {
  final int queueWaitMs;
  final int yuvToRgbMs;
  final int rotateMs;
  final int resizeLetterboxMs;
  final int tensorBuildMs;
  final int inferMs;
  final int postprocessMs;
  final int totalMs;

  const FramePerf({
    this.queueWaitMs = 0,
    this.yuvToRgbMs = 0,
    this.rotateMs = 0,
    this.resizeLetterboxMs = 0,
    this.tensorBuildMs = 0,
    this.inferMs = 0,
    this.postprocessMs = 0,
    this.totalMs = 0,
  });

  factory FramePerf.fromMap(Map<dynamic, dynamic>? json) {
    if (json == null) return const FramePerf();
    return FramePerf(
      queueWaitMs: (json['queueWaitMs'] ?? 0) as int,
      yuvToRgbMs: (json['yuvToRgbMs'] ?? 0) as int,
      rotateMs: (json['rotateMs'] ?? 0) as int,
      resizeLetterboxMs: (json['resizeLetterboxMs'] ?? 0) as int,
      tensorBuildMs: (json['tensorBuildMs'] ?? 0) as int,
      inferMs: (json['inferMs'] ?? 0) as int,
      postprocessMs: (json['postprocessMs'] ?? 0) as int,
      totalMs: (json['totalMs'] ?? 0) as int,
    );
  }

  Map<String, int> toMap() => {
        'queueWaitMs': queueWaitMs,
        'yuvToRgbMs': yuvToRgbMs,
        'rotateMs': rotateMs,
        'resizeLetterboxMs': resizeLetterboxMs,
        'tensorBuildMs': tensorBuildMs,
        'inferMs': inferMs,
        'postprocessMs': postprocessMs,
        'totalMs': totalMs,
      };

  String toPrettyString() {
    return 'Очередь: ${queueWaitMs} мс · '
        'YUV→RGB: ${yuvToRgbMs} мс · '
        'Поворот: ${rotateMs} мс · '
        'Resize: ${resizeLetterboxMs} мс · '
        'Tensor: ${tensorBuildMs} мс · '
        'Infer: ${inferMs} мс · '
        'Post: ${postprocessMs} мс · '
        'Всего: ${totalMs} мс';
  }
}

class DetectionFrameResult {
  final List<DetectionResult> detections;
  final FramePerf perf;

  const DetectionFrameResult({
    required this.detections,
    required this.perf,
  });
}
