import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:developer' as developer;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../api/model_sync_service.dart';
import 'detection_result.dart';

class TfliteDateDetector {
  final ModelSyncService _modelSyncService;

  Isolate? _workerIsolate;
  ReceivePort? _receivePort;
  SendPort? _workerSendPort;
  String? _loadedModelPath;
  int _nextRequestId = 1;
  final Map<int, Completer<DetectionFrameResult>> _pendingRequests = {};

  TfliteDateDetector({ModelSyncService? modelSyncService})
      : _modelSyncService = modelSyncService ?? ModelSyncService();

  bool get isReady => _workerSendPort != null;

  Future<bool> loadLatestModel() async {
    final modelPath = await _modelSyncService.localModelPath();
    if (modelPath == null) {
      close();
      return false;
    }

    if (_loadedModelPath == modelPath && _workerSendPort != null) {
      return true;
    }

    close();

    final file = File(modelPath);
    if (!await file.exists()) {
      return false;
    }

    final receivePort = ReceivePort();
    final readyPortCompleter = Completer<SendPort>();
    final initCompleter = Completer<bool>();

    receivePort.listen((message) {
      if (message is Map) {
        final type = message['type'];
        if (type == 'ready' && message['sendPort'] is SendPort) {
          if (!readyPortCompleter.isCompleted) {
            readyPortCompleter.complete(message['sendPort'] as SendPort);
          }
          return;
        }

        if (type == 'inited') {
          final ok = message['ok'] == true;
          if (!initCompleter.isCompleted) {
            if (ok) {
              initCompleter.complete(true);
            } else {
              initCompleter.completeError(
                message['error']?.toString() ?? 'Не удалось инициализировать worker ИИ.',
              );
            }
          }
          return;
        }

        if (type == 'result') {
          final requestId = message['requestId'] as int?;
          if (requestId == null) return;
          final completer = _pendingRequests.remove(requestId);
          if (completer == null || completer.isCompleted) return;

          if (message['ok'] == true) {
            final raw = (message['detections'] as List?) ?? const [];
            final detections = raw
                .whereType<Map>()
                .map(
                  (item) => DetectionResult(
                    confidence: (item['confidence'] as num).toDouble(),
                    classIndex: (item['classIndex'] as num).toInt(),
                    left: (item['left'] as num).toDouble(),
                    top: (item['top'] as num).toDouble(),
                    right: (item['right'] as num).toDouble(),
                    bottom: (item['bottom'] as num).toDouble(),
                  ),
                )
                .toList(growable: false);
            final perf = FramePerf.fromMap(message['perf'] as Map?);
            developer.log('AI PERF: ${perf.toPrettyString()}', name: 'AI');
            completer.complete(DetectionFrameResult(detections: detections, perf: perf));
          } else {
            completer.completeError(message['error']?.toString() ?? 'Ошибка worker ИИ.');
          }
        }
      }
    });

    final isolate = await Isolate.spawn(_detectorWorkerEntry, receivePort.sendPort);
    final workerSendPort = await readyPortCompleter.future.timeout(const Duration(seconds: 10));

    _workerIsolate = isolate;
    _receivePort = receivePort;
    _workerSendPort = workerSendPort;
    _loadedModelPath = modelPath;

    workerSendPort.send({
      'type': 'init',
      'modelPath': modelPath,
      'threads': 2,
    });

    try {
      return await initCompleter.future.timeout(const Duration(seconds: 20));
    } catch (_) {
      close();
      rethrow;
    }
  }

  Future<DetectionFrameResult> detectFromCameraImage(
    CameraImage cameraImage, {
    required int rotationDegrees,
    bool mirrorHorizontally = false,
  }) async {
    final workerSendPort = _workerSendPort;
    if (workerSendPort == null) {
      return const DetectionFrameResult(detections: [], perf: FramePerf());
    }

    final requestId = _nextRequestId++;
    final completer = Completer<DetectionFrameResult>();
    _pendingRequests[requestId] = completer;

    final planes = cameraImage.planes
        .map(
          (plane) => <String, Object?>{
            'bytes': TransferableTypedData.fromList([plane.bytes]),
            'bytesPerRow': plane.bytesPerRow,
            'bytesPerPixel': plane.bytesPerPixel,
          },
        )
        .toList(growable: false);

    workerSendPort.send({
      'type': 'detect',
      'requestId': requestId,
      'width': cameraImage.width,
      'height': cameraImage.height,
      'formatGroup': cameraImage.format.group.index,
      'rotationDegrees': rotationDegrees,
      'mirrorHorizontally': mirrorHorizontally,
      'planes': planes,
      'submittedAtMicros': DateTime.now().microsecondsSinceEpoch,
    });

    try {
      return await completer.future.timeout(const Duration(seconds: 12));
    } finally {
      _pendingRequests.remove(requestId);
    }
  }

  void close() {
    for (final completer in _pendingRequests.values) {
      if (!completer.isCompleted) {
        completer.complete(const DetectionFrameResult(detections: [], perf: FramePerf()));
      }
    }
    _pendingRequests.clear();

    _workerSendPort?.send({'type': 'dispose'});
    _workerSendPort = null;
    _receivePort?.close();
    _receivePort = null;
    _workerIsolate?.kill(priority: Isolate.immediate);
    _workerIsolate = null;
    _loadedModelPath = null;
  }
}

void _detectorWorkerEntry(SendPort mainSendPort) async {
  final port = ReceivePort();
  mainSendPort.send({'type': 'ready', 'sendPort': port.sendPort});

  Interpreter? interpreter;
  TensorType inputType = TensorType.float32;
  TensorType outputType = TensorType.float32;
  int inputSize = 640;
  double inputScale = 1.0;
  int inputZeroPoint = 0;
  double outputScale = 1.0;
  int outputZeroPoint = 0;

  await for (final message in port) {
    if (message is! Map) continue;
    final type = message['type'];

    if (type == 'dispose') {
      interpreter?.close();
      port.close();
      break;
    }

    if (type == 'init') {
      try {
        interpreter?.close();
        final modelPath = message['modelPath']?.toString();
        if (modelPath == null || modelPath.isEmpty) {
          throw StateError('Путь к модели не передан.');
        }

        final options = InterpreterOptions()..threads = (message['threads'] as int?) ?? 2;
        final created = Interpreter.fromFile(File(modelPath), options: options);
        created.allocateTensors();

        final inputTensor = created.getInputTensor(0);
        final outputTensor = created.getOutputTensor(0);

        inputType = inputTensor.type;
        outputType = outputTensor.type;
        inputSize = inputTensor.shape.length >= 3 ? inputTensor.shape[1] : 640;

        final inputParams = inputTensor.params;
        inputScale = inputParams.scale == 0 ? 1.0 : inputParams.scale;
        inputZeroPoint = inputParams.zeroPoint;

        final outputParams = outputTensor.params;
        outputScale = outputParams.scale == 0 ? 1.0 : outputParams.scale;
        outputZeroPoint = outputParams.zeroPoint;

        interpreter = created;
        mainSendPort.send({'type': 'inited', 'ok': true});
      } catch (e, st) {
        mainSendPort.send({'type': 'inited', 'ok': false, 'error': '$e\n$st'});
      }
      continue;
    }

    if (type == 'detect') {
      final requestId = message['requestId'] as int?;
      if (requestId == null) continue;

      try {
        final activeInterpreter = interpreter;
        if (activeInterpreter == null) {
          throw StateError('Interpreter ещё не инициализирован.');
        }

        final width = message['width'] as int;
        final height = message['height'] as int;
        final formatGroup = message['formatGroup'] as int;
        final rotationDegrees = message['rotationDegrees'] as int;
        final mirrorHorizontally = message['mirrorHorizontally'] == true;
        final submittedAtMicros = message['submittedAtMicros'] as int?;
        final planes = (message['planes'] as List).cast<Map>();

        final totalSw = Stopwatch()..start();
        final perf = <String, int>{
          'queueWaitMs': submittedAtMicros == null
              ? 0
              : ((DateTime.now().microsecondsSinceEpoch - submittedAtMicros) / 1000).round(),
        };

        final cameraImage = _WorkerCameraImage(
          width: width,
          height: height,
          formatGroupIndex: formatGroup,
          planes: planes
              .map(
                (plane) => _WorkerPlane(
                  bytes: (plane['bytes'] as TransferableTypedData).materialize().asUint8List(),
                  bytesPerRow: plane['bytesPerRow'] as int,
                  bytesPerPixel: plane['bytesPerPixel'] as int?,
                ),
              )
              .toList(growable: false),
        );

        final yuvSw = Stopwatch()..start();
        final converted = _cameraImageToImage(cameraImage);
        yuvSw.stop();
        perf['yuvToRgbMs'] = yuvSw.elapsedMilliseconds;
        if (converted == null) {
          mainSendPort.send({
            'type': 'result',
            'requestId': requestId,
            'ok': true,
            'detections': const <Map<String, Object>>[],
          });
          continue;
        }

        final rotateSw = Stopwatch()..start();
        var oriented = _applyRotation(converted, rotationDegrees);
        rotateSw.stop();
        perf['rotateMs'] = rotateSw.elapsedMilliseconds;
        if (mirrorHorizontally) {
          oriented = img.flipHorizontal(oriented);
        }

        final resizeSw = Stopwatch()..start();
        final prep = _letterbox(oriented, inputSize);
        resizeSw.stop();
        perf['resizeLetterboxMs'] = resizeSw.elapsedMilliseconds;
        final outputShape = activeInterpreter.getOutputTensor(0).shape;

        List<Map<String, Object>> detections;
        final tensorSw = Stopwatch()..start();
        if (inputType == TensorType.int8) {
          final inputData = _buildInt8Input(prep.image, inputSize, inputScale, inputZeroPoint);
          tensorSw.stop();
          perf['tensorBuildMs'] = tensorSw.elapsedMilliseconds;

          final output = _createOutputContainerInt(outputShape);
          final inferSw = Stopwatch()..start();
          activeInterpreter.run(inputData, output);
          inferSw.stop();
          perf['inferMs'] = inferSw.elapsedMilliseconds;

          final postSw = Stopwatch()..start();
          detections = _parseOutputInt8(output, prep, inputSize, outputScale, outputZeroPoint);
          postSw.stop();
          perf['postprocessMs'] = postSw.elapsedMilliseconds;
        } else {
          final inputData = _buildFloatInput(prep.image, inputSize);
          tensorSw.stop();
          perf['tensorBuildMs'] = tensorSw.elapsedMilliseconds;

          final output = _createOutputContainerFloat(outputShape);
          final inferSw = Stopwatch()..start();
          activeInterpreter.run(inputData, output);
          inferSw.stop();
          perf['inferMs'] = inferSw.elapsedMilliseconds;

          final postSw = Stopwatch()..start();
          detections = _parseOutputFloat(output, prep, inputSize);
          postSw.stop();
          perf['postprocessMs'] = postSw.elapsedMilliseconds;
        }

        totalSw.stop();
        perf['totalMs'] = totalSw.elapsedMilliseconds;

        mainSendPort.send({
          'type': 'result',
          'requestId': requestId,
          'ok': true,
          'detections': detections,
          'perf': perf,
        });
      } catch (e, st) {
        mainSendPort.send({
          'type': 'result',
          'requestId': requestId,
          'ok': false,
          'error': '$e\n$st',
        });
      }
    }
  }
}

class _WorkerCameraImage {
  final int width;
  final int height;
  final int formatGroupIndex;
  final List<_WorkerPlane> planes;

  const _WorkerCameraImage({
    required this.width,
    required this.height,
    required this.formatGroupIndex,
    required this.planes,
  });
}

class _WorkerPlane {
  final Uint8List bytes;
  final int bytesPerRow;
  final int? bytesPerPixel;

  const _WorkerPlane({
    required this.bytes,
    required this.bytesPerRow,
    required this.bytesPerPixel,
  });
}

img.Image? _cameraImageToImage(_WorkerCameraImage cameraImage) {
  if (cameraImage.formatGroupIndex == ImageFormatGroup.bgra8888.index) {
    return img.Image.fromBytes(
      width: cameraImage.width,
      height: cameraImage.height,
      bytes: cameraImage.planes.first.bytes.buffer,
      order: img.ChannelOrder.bgra,
    );
  }

  if (cameraImage.formatGroupIndex != ImageFormatGroup.yuv420.index) {
    return null;
  }

  final width = cameraImage.width;
  final height = cameraImage.height;
  final yPlane = cameraImage.planes[0];
  final uPlane = cameraImage.planes[1];
  final vPlane = cameraImage.planes[2];

  final image = img.Image(width: width, height: height);

  for (var y = 0; y < height; y++) {
    final uvRow = (y >> 1) * uPlane.bytesPerRow;
    final yRow = y * yPlane.bytesPerRow;

    for (var x = 0; x < width; x++) {
      final uvIndex = uvRow + (x >> 1) * (uPlane.bytesPerPixel ?? 1);
      final index = yRow + x;

      final yp = yPlane.bytes[index];
      final up = uPlane.bytes[uvIndex];
      final vp = vPlane.bytes[uvIndex];

      final r = (yp + 1.402 * (vp - 128)).round();
      final g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128)).round();
      final b = (yp + 1.772 * (up - 128)).round();

      image.setPixelRgb(x, y, _clampColor(r), _clampColor(g), _clampColor(b));
    }
  }

  return image;
}

img.Image _applyRotation(img.Image image, int rotationDegrees) {
  final normalized = ((rotationDegrees % 360) + 360) % 360;
  switch (normalized) {
    case 90:
      return img.copyRotate(image, angle: 90);
    case 180:
      return img.copyRotate(image, angle: 180);
    case 270:
      return img.copyRotate(image, angle: 270);
    default:
      return image;
  }
}

_LetterboxResult _letterbox(img.Image image, int inputSize) {
  final srcWidth = image.width;
  final srcHeight = image.height;
  final scale = math.min(inputSize / srcWidth, inputSize / srcHeight);

  final resizedWidth = math.max(1, (srcWidth * scale).round());
  final resizedHeight = math.max(1, (srcHeight * scale).round());
  final padX = ((inputSize - resizedWidth) / 2).floor();
  final padY = ((inputSize - resizedHeight) / 2).floor();

  final resized = img.copyResize(
    image,
    width: resizedWidth,
    height: resizedHeight,
    interpolation: img.Interpolation.linear,
  );

  final canvas = img.Image(width: inputSize, height: inputSize);
  img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
  img.compositeImage(canvas, resized, dstX: padX, dstY: padY);

  return _LetterboxResult(
    image: canvas,
    originalWidth: srcWidth,
    originalHeight: srcHeight,
    scale: scale,
    padX: padX.toDouble(),
    padY: padY.toDouble(),
  );
}

List<List<List<List<double>>>> _buildFloatInput(img.Image image, int inputSize) {
  final rgbBytes = image.getBytes(order: img.ChannelOrder.rgb);
  var offset = 0;

  return [
    List.generate(
      inputSize,
      (_) => List.generate(
        inputSize,
        (_) {
          final pixel = <double>[
            rgbBytes[offset] / 255.0,
            rgbBytes[offset + 1] / 255.0,
            rgbBytes[offset + 2] / 255.0,
          ];
          offset += 3;
          return pixel;
        },
        growable: false,
      ),
      growable: false,
    ),
  ];
}

List<List<List<List<int>>>> _buildInt8Input(
  img.Image image,
  int inputSize,
  double inputScale,
  int inputZeroPoint,
) {
  final rgbBytes = image.getBytes(order: img.ChannelOrder.rgb);
  var offset = 0;

  return [
    List.generate(
      inputSize,
      (_) => List.generate(
        inputSize,
        (_) {
          final r = _quantizeInput(rgbBytes[offset], inputScale, inputZeroPoint);
          final g = _quantizeInput(rgbBytes[offset + 1], inputScale, inputZeroPoint);
          final b = _quantizeInput(rgbBytes[offset + 2], inputScale, inputZeroPoint);
          offset += 3;
          return <int>[r, g, b];
        },
        growable: false,
      ),
      growable: false,
    ),
  ];
}

int _quantizeInput(int channelValue, double inputScale, int inputZeroPoint) {
  final normalized = channelValue / 255.0;
  final quantized = (normalized / inputScale + inputZeroPoint).round();
  return quantized.clamp(-128, 127);
}

dynamic _createOutputContainerFloat(List<int> shape) {
  if (shape.length == 3) {
    return List.generate(
      shape[0],
      (_) => List.generate(
        shape[1],
        (_) => List<double>.filled(shape[2], 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );
  }

  if (shape.length == 2) {
    return List.generate(
      shape[0],
      (_) => List<double>.filled(shape[1], 0.0, growable: false),
      growable: false,
    );
  }

  throw UnsupportedError('Unsupported float output shape: $shape');
}

dynamic _createOutputContainerInt(List<int> shape) {
  if (shape.length == 3) {
    return List.generate(
      shape[0],
      (_) => List.generate(
        shape[1],
        (_) => List<int>.filled(shape[2], 0, growable: false),
        growable: false,
      ),
      growable: false,
    );
  }

  if (shape.length == 2) {
    return List.generate(
      shape[0],
      (_) => List<int>.filled(shape[1], 0, growable: false),
      growable: false,
    );
  }

  throw UnsupportedError('Unsupported int output shape: $shape');
}

List<Map<String, Object>> _parseOutputFloat(dynamic output, _LetterboxResult prep, int inputSize) {
  const confidenceThreshold = 0.05;
  final rows = _extractFloatRows(output);
  final results = <Map<String, Object>>[];

  for (final row in rows) {
    if (row.length < 6) continue;

    final confidence = row[4];
    if (confidence < confidenceThreshold) {
      continue;
    }

    final left = _mapX(row[0], prep, inputSize);
    final top = _mapY(row[1], prep, inputSize);
    final right = _mapX(row[2], prep, inputSize);
    final bottom = _mapY(row[3], prep, inputSize);

    if (right <= left || bottom <= top) {
      continue;
    }

    results.add({
      'confidence': confidence,
      'classIndex': row[5].round(),
      'left': left,
      'top': top,
      'right': right,
      'bottom': bottom,
    });
  }

  results.sort(
    (a, b) => ((b['confidence'] as num).toDouble()).compareTo((a['confidence'] as num).toDouble()),
  );
  return results.take(10).toList(growable: false);
}

List<Map<String, Object>> _parseOutputInt8(
  dynamic output,
  _LetterboxResult prep,
  int inputSize,
  double outputScale,
  int outputZeroPoint,
) {
  final rows = _extractIntRows(output);
  final dequantized = rows
      .map((row) => row.map((v) => (v - outputZeroPoint) * outputScale).toList(growable: false))
      .toList(growable: false);
  return _parseOutputFloat([dequantized], prep, inputSize);
}

List<List<double>> _extractFloatRows(dynamic output) {
  if (output is List && output.isNotEmpty) {
    final first = output.first;
    if (first is List && first.isNotEmpty && first.first is List) {
      if (first.first is List<double>) {
        return List<List<double>>.from(first);
      }
      return first
          .map<List<double>>((row) => (row as List).map((v) => (v as num).toDouble()).toList(growable: false))
          .toList(growable: false);
    }

    if (first is List && first.length == 6) {
      return output
          .map<List<double>>((row) => (row as List).map((v) => (v as num).toDouble()).toList(growable: false))
          .toList(growable: false);
    }
  }

  throw UnsupportedError('Unsupported float output container');
}

List<List<int>> _extractIntRows(dynamic output) {
  if (output is List && output.isNotEmpty) {
    final first = output.first;
    if (first is List && first.isNotEmpty && first.first is List) {
      return first
          .map<List<int>>((row) => (row as List).map((v) => (v as num).toInt()).toList(growable: false))
          .toList(growable: false);
    }

    if (first is List && first.length == 6) {
      return output
          .map<List<int>>((row) => (row as List).map((v) => (v as num).toInt()).toList(growable: false))
          .toList(growable: false);
    }
  }

  throw UnsupportedError('Unsupported int output container');
}

double _mapX(double rawValue, _LetterboxResult prep, int inputSize) {
  final modelX = rawValue <= 1.2 ? rawValue * inputSize : rawValue;
  final originalX = ((modelX - prep.padX) / prep.scale).clamp(0.0, prep.originalWidth.toDouble());
  return (originalX / prep.originalWidth).clamp(0.0, 1.0);
}

double _mapY(double rawValue, _LetterboxResult prep, int inputSize) {
  final modelY = rawValue <= 1.2 ? rawValue * inputSize : rawValue;
  final originalY = ((modelY - prep.padY) / prep.scale).clamp(0.0, prep.originalHeight.toDouble());
  return (originalY / prep.originalHeight).clamp(0.0, 1.0);
}

int _clampColor(int value) => math.max(0, math.min(255, value));

class _LetterboxResult {
  final img.Image image;
  final int originalWidth;
  final int originalHeight;
  final double scale;
  final double padX;
  final double padY;

  const _LetterboxResult({
    required this.image,
    required this.originalWidth,
    required this.originalHeight,
    required this.scale,
    required this.padX,
    required this.padY,
  });
}
