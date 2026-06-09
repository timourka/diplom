import 'product.dart';

class StoredProduct {
  final int id;
  final int productId;
  final DateTime? manufactureAt;
  final DateTime? expiryAt;
  final DateTime createdAt;
  final Product? product;
  final bool isPendingLocal;

  StoredProduct({
    required this.id,
    required this.productId,
    required this.createdAt,
    this.manufactureAt,
    this.expiryAt,
    this.product,
    this.isPendingLocal = false,
  });

  factory StoredProduct.fromJson(Map<String, dynamic> json) => StoredProduct(
        id: _asInt(json['id']) ?? _asInt(json['localId']) ?? 0,
        productId: _asInt(json['productId']) ?? 0,
        manufactureAt: _parseDate(json['manufactureAt']),
        expiryAt: _parseDate(json['expiryAt']),
        createdAt: _parseDate(json['createdAt']) ?? DateTime.now(),
        product: json['product'] == null ? null : Product.fromJson(Map<String, dynamic>.from(json['product'] as Map)),
        isPendingLocal: json['isPendingLocal'] == true,
      );

  static int? _asInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '');
  }

  static DateTime? _parseDate(dynamic value) {
    final s = value?.toString();
    if (s == null || s.isEmpty) return null;
    return DateTime.tryParse(s);
  }
}
