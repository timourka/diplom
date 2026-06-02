import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class DateReaderResult {
  const DateReaderResult({
    required this.rawText,
    required this.normalizedText,
    required this.confidence,
  });

  final String rawText;
  final String normalizedText;
  final double confidence;

  bool get looksLikeDate {
    final s = normalizedText;
    return RegExp(r'^\d{1,4}[./-]\d{1,2}([./-]\d{1,4})?$').hasMatch(s) ||
        RegExp(r'^\d{6,8}$').hasMatch(s);
  }
}

class DateReaderTflite {
  DateReaderTflite._();

  static final DateReaderTflite instance = DateReaderTflite._();

  Interpreter? _interpreter;
  String _alphabet = '0123456789./-';
  int _blankIndex = 13;
  int _inputWidth = 256;
  int _inputHeight = 48;
  int _timeSteps = 64;
  int _numClasses = 14;

  Future<void> load({
    String modelAsset = 'assets/ai/date_reader.tflite',
    String metaAsset = 'assets/ai/date_reader_meta.json',
  }) async {
    if (_interpreter != null) return;

    final metaRaw = await rootBundle.loadString(metaAsset);
    final meta = jsonDecode(metaRaw) as Map<String, dynamic>;

    _alphabet = (meta['alphabet'] ?? _alphabet).toString();
    _blankIndex = (meta['blank_index'] as num?)?.toInt() ?? _alphabet.length;
    _inputWidth = (meta['input_width'] as num?)?.toInt() ?? _inputWidth;
    _inputHeight = (meta['input_height'] as num?)?.toInt() ?? _inputHeight;
    _timeSteps = (meta['output_time_steps'] as num?)?.toInt() ?? _timeSteps;
    _numClasses = (meta['num_classes'] as num?)?.toInt() ?? (_alphabet.length + 1);

    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(modelAsset, options: options);

    final outShape = _interpreter!.getOutputTensor(0).shape;
    if (outShape.length == 3) {
      _timeSteps = outShape[1];
      _numClasses = outShape[2];
    }
  }

  Future<DateReaderResult?> recognizeFile(File imageFile) async {
    await load();
    final bytes = await imageFile.readAsBytes();
    return recognizeBytes(bytes);
  }

  Future<DateReaderResult?> recognizeBytes(Uint8List bytes) async {
    await load();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) return null;

    final input = _preprocess(decoded);
    final output = List.generate(
      1,
      (_) => List.generate(
        _timeSteps,
        (_) => List<double>.filled(_numClasses, 0.0),
        growable: false,
      ),
      growable: false,
    );

    _interpreter!.run(input, output);

    final decodedText = _ctcGreedyDecode(output[0]);
    final normalized = _normalizeDateText(decodedText);
    final confidence = _confidence(output[0]);

    if (normalized.isEmpty) return null;
    return DateReaderResult(
      rawText: decodedText,
      normalizedText: normalized,
      confidence: confidence,
    );
  }

  List<List<List<List<double>>>> _preprocess(img.Image source) {
    final gray = img.grayscale(source);
    final w = gray.width;
    final h = gray.height;
    final scale = math.min(_inputWidth / math.max(1, w), _inputHeight / math.max(1, h));
    final nw = math.max(1, (w * scale).floor());
    final nh = math.max(1, (h * scale).floor());

    final resized = img.copyResize(
      gray,
      width: nw,
      height: nh,
      interpolation: img.Interpolation.linear,
    );

    final canvas = img.Image(width: _inputWidth, height: _inputHeight);
    img.fill(canvas, color: img.ColorRgb8(255, 255, 255));
    final yOffset = ((_inputHeight - nh) / 2).floor();
    img.compositeImage(canvas, resized, dstX: 0, dstY: yOffset);

    return List.generate(
      1,
      (_) => List.generate(
        _inputHeight,
        (y) => List.generate(
          _inputWidth,
          (x) {
            final p = canvas.getPixel(x, y);
            return <double>[p.r / 255.0];
          },
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );
  }

  String _ctcGreedyDecode(List<List<double>> logits) {
    final out = StringBuffer();
    int? prev;

    for (final step in logits) {
      var bestIndex = 0;
      var bestValue = step[0];
      for (var i = 1; i < step.length; i++) {
        if (step[i] > bestValue) {
          bestValue = step[i];
          bestIndex = i;
        }
      }

      if (bestIndex != prev && bestIndex != _blankIndex && bestIndex >= 0 && bestIndex < _alphabet.length) {
        out.write(_alphabet[bestIndex]);
      }
      prev = bestIndex;
    }

    return out.toString();
  }

  double _confidence(List<List<double>> logits) {
    var sum = 0.0;
    var count = 0;
    int? prev;

    for (final step in logits) {
      var bestIndex = 0;
      var bestValue = step[0];
      for (var i = 1; i < step.length; i++) {
        if (step[i] > bestValue) {
          bestValue = step[i];
          bestIndex = i;
        }
      }

      if (bestIndex != prev && bestIndex != _blankIndex) {
        sum += _softmaxMax(step, bestValue);
        count++;
      }
      prev = bestIndex;
    }

    return count == 0 ? 0.0 : sum / count;
  }

  double _softmaxMax(List<double> values, double maxValue) {
    var denom = 0.0;
    for (final v in values) {
      denom += math.exp(v - maxValue);
    }
    return denom <= 0 ? 0.0 : 1.0 / denom;
  }

  String _normalizeDateText(String text) {
    var s = text.trim();
    s = s.replaceAll(' ', '.');
    s = s.replaceAll('|', '/');
    s = s.replaceAll('\\', '/');
    s = s.replaceAll(RegExp(r'[^0-9./-]'), '');
    s = s.replaceAll(RegExp(r'[.]{2,}'), '.');
    s = s.replaceAll(RegExp(r'[/]{2,}'), '/');
    s = s.replaceAll(RegExp(r'[-]{2,}'), '-');
    return s;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
