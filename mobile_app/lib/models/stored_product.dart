import 'product.dart';

class StoredProduct {
  final int id;
  final int productId;
  final DateTime? manufactureAt;
  final DateTime? expiryAt;
  final DateTime createdAt;
  final Product? product;

  StoredProduct({
    required this.id,
    required this.productId,
    required this.createdAt,
    this.manufactureAt,
    this.expiryAt,
    this.product,
  });

  factory StoredProduct.fromJson(Map<String, dynamic> json) => StoredProduct(
        id: json['id'] as int,
        productId: json['productId'] as int,
        manufactureAt: json['manufactureAt'] == null ? null : DateTime.parse(json['manufactureAt']),
        expiryAt: json['expiryAt'] == null ? null : DateTime.parse(json['expiryAt']),
        createdAt: DateTime.parse(json['createdAt']),
        product: json['product'] == null ? null : Product.fromJson(json['product']),
      );
}
