import 'dart:io';
import 'dart:math' as math;

import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:image/image.dart' as img;

class DateOcrResult {
  final DateTime? date;
  final String rawText;
  final String? matchedText;

  const DateOcrResult({
    required this.date,
    required this.rawText,
    this.matchedText,
  });

  bool get hasDate => date != null;
}

class DateOcrService {
  final TextRecognizer _recognizer = TextRecognizer(script: TextRecognitionScript.latin);

  Future<DateOcrResult> recognizeDateFromImagePath(String imagePath) async {
    final variantFiles = await _prepareOcrVariants(imagePath);
    final rawParts = <String>[];

    try {
      for (final variant in variantFiles) {
        try {
          final inputImage = InputImage.fromFilePath(variant.path);
          final recognizedText = await _recognizer.processImage(inputImage);
          final text = recognizedText.text.trim();
          if (text.isNotEmpty) {
            rawParts.add(text);
          }
        } catch (_) {
          // Один неудачный вариант предобработки не должен ломать OCR целиком.
        }
      }
    } finally {
      for (final variant in variantFiles) {
        if (!variant.deleteAfterUse) continue;
        try {
          final file = File(variant.path);
          if (await file.exists()) await file.delete();
        } catch (_) {}
      }
    }

    final rawText = rawParts.join('\n');
    final parsed = DateTextParser.parse(rawText);

    // Для отладки: если дата глазами видна, но не распознана, здесь будет понятно,
    // что именно вернул ML Kit после всех вариантов предобработки.
    // ignore: avoid_print
    print('[OCR] raw="$rawText" matched="${parsed?.matchedText}" date="${parsed?.date}"');

    return DateOcrResult(
      date: parsed?.date,
      rawText: rawText,
      matchedText: parsed?.matchedText,
    );
  }

  Future<List<_OcrVariantFile>> _prepareOcrVariants(String imagePath) async {
    final result = <_OcrVariantFile>[_OcrVariantFile(imagePath, deleteAfterUse: false)];

    final bytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) return result;

    final oriented = img.bakeOrientation(decoded);
    final scaled = _scaleForOcr(oriented);
    final paddedOriginal = _addWhitePadding(scaled, 28);

    final gray = img.grayscale(scaled);
    final contrast = img.adjustColor(gray, contrast: 1.75, brightness: 0.04, saturation: 0.0);
    final paddedContrast = _addWhitePadding(contrast, 28);

    final binary = _binarizeDarkText(scaled);
    final paddedBinary = _addWhitePadding(binary, 28);

    final dilatedLight = _dilateDarkPixels(binary, radiusX: 1, radiusY: 1);
    final paddedDilatedLight = _addWhitePadding(dilatedLight, 28);

    // Для точечно-матричной печати полезно слегка соединять точки по горизонтали.
    final dilatedWide = _dilateDarkPixels(binary, radiusX: 2, radiusY: 1);
    final paddedDilatedWide = _addWhitePadding(dilatedWide, 28);

    final inverted = img.invert(img.Image.from(paddedDilatedLight));

    result.add(await _writeVariant(imagePath, 'ocr_scaled', paddedOriginal));
    result.add(await _writeVariant(imagePath, 'ocr_contrast', paddedContrast));
    result.add(await _writeVariant(imagePath, 'ocr_binary', paddedBinary));
    result.add(await _writeVariant(imagePath, 'ocr_dilate', paddedDilatedLight));
    result.add(await _writeVariant(imagePath, 'ocr_dilate_wide', paddedDilatedWide));
    result.add(await _writeVariant(imagePath, 'ocr_invert', inverted));

    return result;
  }

  img.Image _scaleForOcr(img.Image source) {
    if (source.width <= 0 || source.height <= 0) return source;

    const targetMinHeight = 260;
    const targetMaxWidth = 2200;

    var scale = targetMinHeight / source.height;
    if (scale < 1.0) scale = 1.0;

    var width = (source.width * scale).round();
    var height = (source.height * scale).round();

    if (width > targetMaxWidth) {
      final downScale = targetMaxWidth / width;
      width = targetMaxWidth;
      height = math.max(1, (height * downScale).round());
    }

    if (width == source.width && height == source.height) {
      return img.Image.from(source);
    }

    return img.copyResize(
      source,
      width: width,
      height: height,
      interpolation: img.Interpolation.cubic,
    );
  }

  img.Image _addWhitePadding(img.Image source, int padding) {
    final out = img.Image(width: source.width + padding * 2, height: source.height + padding * 2);
    img.fill(out, color: img.ColorRgb8(255, 255, 255));

    for (var y = 0; y < source.height; y++) {
      for (var x = 0; x < source.width; x++) {
        out.setPixel(x + padding, y + padding, source.getPixel(x, y));
      }
    }

    return out;
  }

  img.Image _binarizeDarkText(img.Image source) {
    final gray = img.grayscale(img.Image.from(source));
    final threshold = _otsuThreshold(gray);
    final out = img.Image(width: gray.width, height: gray.height);

    for (var y = 0; y < gray.height; y++) {
      for (var x = 0; x < gray.width; x++) {
        final p = gray.getPixel(x, y);
        final value = p.r.toInt() <= threshold ? 0 : 255;
        out.setPixelRgb(x, y, value, value, value);
      }
    }

    return out;
  }

  int _otsuThreshold(img.Image gray) {
    final hist = List<int>.filled(256, 0);
    var total = 0;

    for (var y = 0; y < gray.height; y++) {
      for (var x = 0; x < gray.width; x++) {
        final v = gray.getPixel(x, y).r.round().clamp(0, 255);
        hist[v]++;
        total++;
      }
    }

    var sum = 0.0;
    for (var i = 0; i < 256; i++) {
      sum += i * hist[i];
    }

    var sumB = 0.0;
    var wB = 0;
    var maxBetween = -1.0;
    var threshold = 128;

    for (var i = 0; i < 256; i++) {
      wB += hist[i];
      if (wB == 0) continue;

      final wF = total - wB;
      if (wF == 0) break;

      sumB += i * hist[i];
      final mB = sumB / wB;
      final mF = (sum - sumB) / wF;
      final between = wB * wF * math.pow(mB - mF, 2).toDouble();

      if (between > maxBetween) {
        maxBetween = between;
        threshold = i;
      }
    }

    // На цветных этикетках Otsu иногда делает порог слишком светлым и теряет точки.
    // Немного смещаем порог вниз, чтобы оставить только тёмные элементы маркировки.
    return (threshold - 10).clamp(50, 210);
  }

  img.Image _dilateDarkPixels(img.Image binary, {required int radiusX, required int radiusY}) {
    final out = img.Image(width: binary.width, height: binary.height);
    img.fill(out, color: img.ColorRgb8(255, 255, 255));

    for (var y = 0; y < binary.height; y++) {
      for (var x = 0; x < binary.width; x++) {
        var hasDarkNeighbor = false;

        for (var dy = -radiusY; dy <= radiusY && !hasDarkNeighbor; dy++) {
          final yy = y + dy;
          if (yy < 0 || yy >= binary.height) continue;

          for (var dx = -radiusX; dx <= radiusX; dx++) {
            final xx = x + dx;
            if (xx < 0 || xx >= binary.width) continue;

            if (binary.getPixel(xx, yy).r < 128) {
              hasDarkNeighbor = true;
              break;
            }
          }
        }

        final value = hasDarkNeighbor ? 0 : 255;
        out.setPixelRgb(x, y, value, value, value);
      }
    }

    return out;
  }

  Future<_OcrVariantFile> _writeVariant(String originalPath, String suffix, img.Image image) async {
    final path = '${originalPath}_$suffix.png';
    final file = File(path);
    await file.writeAsBytes(img.encodePng(image), flush: true);
    return _OcrVariantFile(path, deleteAfterUse: true);
  }

  void close() {
    _recognizer.close();
  }
}

class _OcrVariantFile {
  final String path;
  final bool deleteAfterUse;

  const _OcrVariantFile(this.path, {required this.deleteAfterUse});
}

class ParsedDateText {
  final DateTime date;
  final String matchedText;

  const ParsedDateText({required this.date, required this.matchedText});
}

class DateTextParser {
  static ParsedDateText? parse(String raw) {
    final normalized = _normalize(raw);
    final candidates = <ParsedDateText>[];

    // yyyy.mm.dd / dd.mm.yyyy / yy.mm.dd / dd.mm.yy
    final separated = RegExp(
      r'(?<!\d)(\d{1,4})\s*[.\-/\\]\s*(\d{1,2})\s*[.\-/\\]\s*(\d{1,4})(?!\d)',
      caseSensitive: false,
    );
    for (final match in separated.allMatches(normalized)) {
      final parsed = _parseThreeParts(
        match.group(1)!,
        match.group(2)!,
        match.group(3)!,
        match.group(0)!,
      );
      if (parsed != null) candidates.add(parsed);
    }

    // Частый случай после OCR: лишний префикс партии перед датой, например 25/02.12.2025.
    final trailingDate = RegExp(
      r'(?:^|[^\d])(?:\d{1,4}[.\-/\\])?(\d{1,2})\s*[.\-/\\]\s*(\d{1,2})\s*[.\-/\\]\s*((?:20)?\d{2})(?!\d)',
      caseSensitive: false,
    );
    for (final match in trailingDate.allMatches(normalized)) {
      final parsed = _makeDate(
        _normalizeYear(int.tryParse(match.group(3)!)),
        int.tryParse(match.group(2)!),
        int.tryParse(match.group(1)!),
        match.group(0)!,
      );
      if (parsed != null) candidates.add(parsed);
    }

    // dd mm yyyy / yyyy mm dd after OCR inserts spaces instead of separators.
    final spaced = RegExp(r'(?<!\d)(\d{1,4})\s+(\d{1,2})\s+(\d{1,4})(?!\d)');
    for (final match in spaced.allMatches(normalized)) {
      final parsed = _parseThreeParts(
        match.group(1)!,
        match.group(2)!,
        match.group(3)!,
        match.group(0)!,
      );
      if (parsed != null) candidates.add(parsed);
    }

    // yyyymmdd / ddmmyyyy compact stamps.
    final compact8 = RegExp(r'(?<!\d)(\d{8})(?!\d)');
    for (final match in compact8.allMatches(normalized)) {
      final token = match.group(1)!;
      final parsedYearFirst = _makeDate(
        int.tryParse(token.substring(0, 4)),
        int.tryParse(token.substring(4, 6)),
        int.tryParse(token.substring(6, 8)),
        match.group(0)!,
      );
      if (parsedYearFirst != null) candidates.add(parsedYearFirst);

      final parsedDayFirst = _makeDate(
        int.tryParse(token.substring(4, 8)),
        int.tryParse(token.substring(2, 4)),
        int.tryParse(token.substring(0, 2)),
        match.group(0)!,
      );
      if (parsedDayFirst != null) candidates.add(parsedDayFirst);
    }

    // mm.yyyy / mm/yyyy. Для срока годности без дня берём последний день месяца.
    final monthYear = RegExp(r'(?<!\d)(\d{1,2})\s*[.\-/\\]\s*((?:20)?\d{2})(?!\d)');
    for (final match in monthYear.allMatches(normalized)) {
      final month = int.tryParse(match.group(1)!);
      final year = _normalizeYear(int.tryParse(match.group(2)!));
      if (year == null || month == null || month < 1 || month > 12) continue;
      final day = DateTime(year, month + 1, 0).day;
      final parsed = _makeDate(year, month, day, match.group(0)!);
      if (parsed != null) candidates.add(parsed);
    }

    if (candidates.isEmpty) return null;

    candidates.sort((a, b) => _scoreDate(b, normalized).compareTo(_scoreDate(a, normalized)));
    return candidates.first;
  }

  static String _normalize(String raw) {
    return raw
        .toUpperCase()
        .replaceAll('О', '0')
        .replaceAll('O', '0')
        .replaceAll('Q', '0')
        .replaceAll('D', '0')
        .replaceAll('I', '1')
        .replaceAll('L', '1')
        .replaceAll('|', '1')
        .replaceAll('S', '5')
        .replaceAll('Б', '6')
        .replaceAll('З', '3')
        .replaceAll('В', '8')
        .replaceAll('B', '8')
        .replaceAll('—', '-')
        .replaceAll('–', '-')
        .replaceAll(',', '.')
        .replaceAll(':', '.')
        .replaceAll(RegExp(r'[^0-9A-ZА-Я.\-/\\\s]'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  static ParsedDateText? _parseThreeParts(String aRaw, String bRaw, String cRaw, String matchedText) {
    final a = int.tryParse(aRaw);
    final b = int.tryParse(bRaw);
    final c = int.tryParse(cRaw);
    if (a == null || b == null || c == null) return null;

    // YYYY-MM-DD
    if (aRaw.length == 4 || a > 31) {
      return _makeDate(_normalizeYear(a), b, c, matchedText);
    }

    // DD-MM-YYYY
    if (cRaw.length == 4 || c > 31) {
      return _makeDate(_normalizeYear(c), b, a, matchedText);
    }

    // YY-MM-DD, если первый блок явно похож на год.
    if (a > 31 || aRaw.length == 2 && b <= 12 && c <= 31 && a >= 20) {
      return _makeDate(_normalizeYear(a), b, c, matchedText);
    }

    // DD-MM-YY. Для русскоязычных упаковок это самый частый вариант.
    return _makeDate(_normalizeYear(c), b, a, matchedText);
  }

  static int? _normalizeYear(int? year) {
    if (year == null) return null;
    if (year < 0) return null;
    if (year < 100) {
      return year >= 70 ? 1900 + year : 2000 + year;
    }
    return year;
  }

  static ParsedDateText? _makeDate(int? year, int? month, int? day, String matchedText) {
    if (year == null || month == null || day == null) return null;
    if (year < 2000 || year > 2099) return null;
    if (month < 1 || month > 12) return null;
    if (day < 1 || day > 31) return null;

    final date = DateTime(year, month, day);
    if (date.year != year || date.month != month || date.day != day) return null;
    return ParsedDateText(date: date, matchedText: matchedText.trim());
  }

  static double _scoreDate(ParsedDateText candidate, String fullText) {
    var score = 0.0;
    final match = candidate.matchedText.toUpperCase();
    final idx = fullText.indexOf(match);
    final before = idx <= 0 ? '' : fullText.substring(math.max(0, idx - 20), idx);
    final after = idx < 0 ? '' : fullText.substring(idx, math.min(fullText.length, idx + match.length + 20));
    final context = '$before $after';

    if (RegExp(r'EXP|EXPIR|BEST|BEFORE|ГОДЕН|ГОДН|СРОК|ДО').hasMatch(context)) score += 5;
    if (RegExp(r'PROD|MFG|ИЗГОТ|ПРОИЗВ').hasMatch(context)) score += 2;
    if (candidate.date.year >= 2020) score += 1;
    if (candidate.date.isAfter(DateTime.now().subtract(const Duration(days: 365 * 3)))) score += 1;
    if (candidate.matchedText.contains(RegExp(r'20\d{2}'))) score += 2;
    if (candidate.matchedText.length >= 8) score += 0.5;
    return score;
  }
}
