class Product {
  final int id;
  final String name;
  final String? manufacturer;
  final String? barcode;

  Product({required this.id, required this.name, this.manufacturer, this.barcode});

  factory Product.fromJson(Map<String, dynamic> json) => Product(
        id: _asInt(json['id']) ?? 0,
        name: (json['name'] ?? '').toString(),
        manufacturer: json['manufacturer']?.toString(),
        barcode: json['barcode']?.toString(),
      );

  static int? _asInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '');
  }
}
